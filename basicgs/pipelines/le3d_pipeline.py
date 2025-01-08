import json
from collections import OrderedDict
from copy import deepcopy

import torch
from basicgs.pipelines import lr_scheduler
from basicgs.utils.logger import get_root_logger
from basicgs.utils.raw_util import finish_isp
from basicgs.utils.registry import PIPE_REGISTRY
from basicgs.pipelines.gs_base_pipeline import GSBasePipeline

def parse_cfa_info(pattern_array):
    """Convert CFA pattern array to DNG format strings.

    Args:
        pattern_array: List[List[int]], e.g. [[0,1], [1,2]]

    Returns:
        tuple: (pattern_dim, pattern_str)
            pattern_dim: str, e.g. "2 2"
            pattern_str: str, e.g. "[Red,Green][Green,Blue]"
    """
    cfa_dict = {
        0: 'Red',
        1: 'Green',
        2: 'Blue',
        3: 'Green'
    }

    height = len(pattern_array)
    width = len(pattern_array[0]) if height > 0 else 0
    pattern_dim = f"{width} {height}"

    pattern_str = ""
    for row in pattern_array:
        pattern_str += "["
        pattern_str += ",".join(cfa_dict[x] for x in row)
        pattern_str += "]"

    pattern_str_2 = ""
    for row in pattern_array:
        for x in row:
            pattern_str_2 += f"{x} "
    pattern_str_2 = pattern_str_2.strip()

    return pattern_dim, pattern_str, pattern_str_2

@PIPE_REGISTRY.register()
class LE3DPipeline(GSBasePipeline):
    def __init__(self, opt, train_loader):
        self.multi_exposure_training = opt['train'].get('multi_exposure_training', False)
        self.other_isp_meta = opt['val'].get('other_isp_meta', {})
        super().__init__(opt, train_loader)
        # save a meta data for visulize
        with torch.no_grad():
            sampled_cam = train_loader[0]['camera']
            sampled_meta_data = sampled_cam.meta_data
            meta_data_dict = {
                'WhiteLevel': int(sampled_meta_data[0].item()),
                'BlackLevel': int(sampled_meta_data[1].item()),
                'AsShotNeutral': ' '.join([str(x) for x in sampled_meta_data[2].numpy().tolist()]),
                'ColorMatrix2': ' '.join([str(x) for x in sampled_meta_data[3].numpy().flatten().tolist()]),
                'ShutterSpeed': f'{sampled_meta_data[4].item()}',
                'NoiseProfile': ' '.join([str(x) for x in sampled_meta_data[5]]),
            }
            if len(sampled_meta_data) > 6:
                pattern_dim, pattern_str, pattern_str_2 = parse_cfa_info(sampled_meta_data[6])
                meta_data_dict['CFARepeatPatternDim'] = pattern_dim
                meta_data_dict['CFAPattern2'] = pattern_str_2
            else:
                meta_data_dict['CFARepeatPatternDim'] = '2 2'
                meta_data_dict['CFAPattern2'] = '0 1 1 2'

        with open(f'{self.opt["path"]["experiments_root"]}/meta_data.json', 'w') as f:
            json.dump(meta_data_dict, f, indent=4)

    def feed_data(self, data):
        if self.multi_exposure_training:
            self.ratio = data['ratio'].to(self.device)
            self.shutter = data['shutter']
        super().feed_data(data)

    def setup_optimizer_color_mlp(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.color_mlp.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_color_mlp'].pop('type')
        self.optimizer_color_mlp = self.get_optimizer(optim_type, optim_params, **train_opt['optim_color_mlp'])
        self.optimizer_color_mlp.name = 'color_mlp'
        self.optimizers.append(self.optimizer_color_mlp)

    def setup_scheduler_color_mlp(self):
        """Set final net schedulers."""
        train_opt = self.opt['train']
        scheduler_type = train_opt['scheduler_color_mlp'].pop('type')
        if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
            self.schedulers.append(lr_scheduler.MultiStepRestartLR(self.optimizer_color_mlp, **train_opt['scheduler_color_mlp']))
        elif scheduler_type == 'CosineAnnealingRestartLR':
            self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(self.optimizer_color_mlp, **train_opt['scheduler_color_mlp']))
        elif scheduler_type == 'ExponentialLR':
            self.schedulers.append(lr_scheduler.ExponentialLR(self.optimizer_color_mlp, **train_opt['scheduler_color_mlp']))
        else:
            raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def setup_optim_exp(self, params):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in params.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} of `exposure_correction_params` will not be optimized.')

        if len(optim_params) == 0:
            logger = get_root_logger()
            logger.warning(f'No need to optimize multi exposure, scene only contain one exposure!')
            return

        optim_type = train_opt['optim_exp'].pop('type')
        self.optimizer_exp = self.get_optimizer(optim_type, optim_params, **train_opt['optim_exp'])
        self.optimizer_exp.name = 'exp'
        self.optimizers.append(self.optimizer_exp)

    def setup_scheduler_exp(self):
        train_opt = deepcopy(self.opt['train'])
        """Set up schedulers."""
        if train_opt.get('scheduler_exp') is not None and hasattr(self, 'optimizer_exp'):
            scheduler_type = train_opt['scheduler_exp'].pop('type')
            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                self.schedulers.append(lr_scheduler.MultiStepRestartLR(self.optimizer_exp, **train_opt['scheduler_exp']))
            elif scheduler_type == 'CosineAnnealingRestartLR':
                self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(self.optimizer_exp, **train_opt['scheduler_exp']))
            elif scheduler_type == 'ExponentialLR':
                self.schedulers.append(lr_scheduler.ExponentialLR(self.optimizer_exp, **train_opt['scheduler_exp']))
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def init_training_settings(self):
        super().init_training_settings()
        self.setup_optimizer_color_mlp()
        self.setup_scheduler_color_mlp()
        if self.multi_exposure_training:
            self.exposure_correction_from = self.net_g.densify_from_iter
            self.setup_optim_exp(self.net_g.exposure_correction_params)
            self.setup_scheduler_exp()


    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_color_mlp.zero_grad()
        if self.multi_exposure_training and hasattr(self, 'optimizer_exp'):
            self.optimizer_exp.zero_grad()

        self.output_pkg = self.net_g(self.curr_camera)
        self.output = self.output_pkg['render']

        ## exposure correction
        if self.multi_exposure_training and current_iter > self.exposure_correction_from:
            exposure_correction = self.net_g.get_exposure_correction_param(self.shutter) * self.ratio
            self.output =  exposure_correction * self.output

        self.gt = self.gt.unsqueeze(0)
        self.output = self.output.unsqueeze(0)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pixs:
            l_pix = 0
            for cri_pix in self.cri_pixs:
                curr_l_pix = cri_pix(self.output, self.gt)
                curr_l_name = getattr(cri_pix, 'name', 'unspecified')
                l_pix += curr_l_pix
                loss_dict[f'l_pix_{curr_l_name}'] = curr_l_pix
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        if self.gs_regs:
            l_gs_regs = 0
            for reg in self.gs_regs:
                curr_reg = reg(self.net_g, self.curr_camera, self.output_pkg, current_iter)
                curr_reg_name = getattr(reg, 'name', 'unspecified')
                l_gs_regs += curr_reg
                loss_dict[f'reg_{curr_reg_name}'] = curr_reg
            l_total += l_gs_regs
            loss_dict['reg'] = l_gs_regs

        if self.multi_exposure_training:
            params = torch.stack(list(self.net_g.exposure_correction_params.parameters()))
            loss_dict['mean_exp_c'] = params.mean()
        if hasattr(self.net_g, '_xyz'):
            loss_dict['count'] = torch.tensor(self.net_g._xyz.shape[0]).float()

        l_total.backward()

        self.net_g.finalize_iter(current_iter, self.output_pkg, self.cameras_extent, self.optimizer_g)

        self.optimizer_g.step()
        self.optimizer_color_mlp.step()
        if self.multi_exposure_training and hasattr(self, 'optimizer_exp'):
            self.optimizer_exp.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # save mem
        del self.curr_camera
        torch.cuda.empty_cache()

    def test(self):
        super().test()
        if self.multi_exposure_training:
            self.output = self.net_g.get_exposure_correction_param(self.shutter) * \
                self.ratio * self.output

    def get_current_visuals(self):
        out_dict = super().get_current_visuals()
        if self.other_isp_meta is not None:
            out_dict['gt']     = finish_isp(self.gt.detach(), self.curr_camera.meta_data,
                                            **self.other_isp_meta).cpu()
            out_dict['result'] = finish_isp(self.output.detach(), self.curr_camera.meta_data,
                                            **self.other_isp_meta).cpu()
        return out_dict
