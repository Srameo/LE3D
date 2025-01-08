import os
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from copy import deepcopy

from basicgs.gaussians import build_gaussians
from basicgs.gaussians.util.render import render_full_package
from basicgs.losses import build_loss
from basicgs.metrics import calculate_metric
from basicgs.utils import get_root_logger, imwrite, tensor2img
from basicgs.utils.registry import PIPE_REGISTRY
from basicgs.viewers.utils import depth_naninf_to_red
from .base_pipeline import BasePipeline

@PIPE_REGISTRY.register()
class GSBasePipeline(BasePipeline):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt, train_loader):
        super(GSBasePipeline, self).__init__(opt, train_loader)

        # define network
        net_g_opt = deepcopy(opt['network_g'])
        if net_g_opt.get('train_dataset'):
            net_g_opt['train_dataset'] = self.train_loader
        self.net_g = build_gaussians(net_g_opt)
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        self.first_save = True
        self.save_training_state_for_debug = self.opt['logger'].get('save_training_state_for_debug', False)
        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt') or train_opt.get('pixel_opts'): # multiple pixel losses: ssim and l2
            assert not train_opt.get('pixel_opt') or not train_opt.get('pixel_opts'), 'Only one of pixel_opt or pixel_opts should be provided.'
            if train_opt.get('pixel_opt'):
                pixel_opts = [train_opt['pixel_opt']]
            else:
                pixel_opts = train_opt['pixel_opts']
            self.cri_pixs = []
            for pixel_opt in pixel_opts:
                self.cri_pixs.append(build_loss(pixel_opt).to(self.device))
        else:
            self.cri_pixs = None

        if train_opt.get('reg_opt') or train_opt.get('reg_opts'):
            assert not train_opt.get('reg_opt') or not train_opt.get('reg_opts'), 'Only one of reg_opt or reg_opts should be provided.'
            if train_opt.get('reg_opt'):
                reg_opts = [train_opt['reg_opt']]
            else:
                reg_opts = train_opt['reg_opts']
            self.gs_regs = []
            for reg_opt in reg_opts:
                reg_opt['net_g'] = self.net_g
                self.gs_regs.append(build_loss(reg_opt).to(self.device))
        else:
            self.gs_regs = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pixs is None:
            raise ValueError('Pixel losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        # set up spatial lr scale
        self.spatial_lr_scale = self.train_loader.nerf_normalization['radius']
        for param_group in self.optimizer_g.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = param_group['lr'] * self.spatial_lr_scale

        self.setup_schedulers()
        for scheduler in self.schedulers:
            scheduler.lr_scale = self.spatial_lr_scale

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        params = self.net_g.fetch_parameters_for_optimizer()
        detailed_lr = train_opt['optim_g'].pop('detailed_lr')
        for k, v in params.items():
            if v['params'][0].requires_grad:
                v['lr'] = detailed_lr[k]
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizer_g.name = 'main'
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.curr_camera = data['camera'].to(self.device)
        self.gt = self.curr_camera.original_image
        self.cameras_extent = data['cameras_extent'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output_pkg = self.net_g(self.curr_camera)
        self.output = self.output_pkg['render']

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

        if hasattr(self.net_g, '_xyz'):
            loss_dict['count'] = torch.tensor(self.net_g._xyz.shape[0]).float()

        l_total.backward()

        self.net_g.finalize_iter(current_iter, self.output_pkg, self.cameras_extent, self.optimizer_g)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        # save mem
        del self.curr_camera
        torch.cuda.empty_cache()

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output_pkg = self.net_g(self.curr_camera)
            self.output = self.output_pkg['render']
            if self.opt['val'].get('render_depth', False):
                full_pkg = render_full_package(self.net_g, self.curr_camera)
                self.output_depth = full_pkg['depth'] / (full_pkg['final_opacity'] + 1.0e-3)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', True)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['camera'].image_name))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            out_img = tensor2img([visuals['result']])
            metric_data['img'] = out_img
            gt_img = tensor2img([visuals['gt']])
            metric_data['img2'] = gt_img

            if visuals.get('depth', None) is not None:
                depth = tensor2img([visuals['depth']])
            else:
                depth = None

            # tentative for out of GPU memory
            del self.curr_camera
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                save_as = self.opt['val'].get('save_as', 'jpg')
                if self.opt['is_train']:
                    img_name_pp = f'{img_name}_{current_iter}'
                else:
                    img_name_pp = img_name
                if self.opt['val'].get('suffix', None):
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                f'{img_name_pp}_{self.opt["val"]["suffix"]}.{save_as}')
                else:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                f'{img_name_pp}.{save_as}')
                if depth is not None:
                    save_depth_path = osp.join(self.opt['path']['visualization'], dataset_name, img_name,
                                                f'{img_name_pp}_depth.{save_as}')
                    imwrite(depth, save_depth_path)
                imwrite(out_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'output_depth'):
            out_dict['depth'] = depth_naninf_to_red(self.output_depth / self.output_depth.max()).detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        if self.save_training_state_for_debug:
            self.save_training_state(epoch, current_iter)
        elif self.first_save:
            with open(osp.join(self.opt['path']['training_states'], 'readme.txt'), 'w') as f:
                f.write('Currently not support save training state.')
            self.first_save = False
        # self.save_training_state(epoch, current_iter)

    def save_network(self, net, net_label, current_iter, param_key='params'):
        if self.opt['logger'].get('save_in_ply', True):
            assert hasattr(self.net_g, 'save_ply'), 'Gaussians should have method save_ply!'
            if current_iter == -1:
                current_iter = 'latest'
            save_filename = f'{net_label}_{current_iter}.ply'
            save_path = os.path.join(self.opt['path']['models'], save_filename)
            self.net_g.save_ply(save_path)
            logger = get_root_logger()
            logger.info(f'Save Gaussians to ply file: {save_path}')
            return
        return super().save_network(net, net_label, current_iter, param_key)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        if load_path.endswith('.ply'):
            assert hasattr(self.net_g, 'load_ply'), 'Gaussians should have method load_ply!'
            self.net_g.load_ply(load_path)
            logger = get_root_logger()
            logger.info(f'Loaded Gaussians from ply file: {load_path}')
            return
        return super().load_network(net, load_path, strict, param_key)
