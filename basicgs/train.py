from copy import deepcopy
import datetime
import logging
import math
from numbers import Number
import time
import torch
from os import path as osp

from basicgs.data import build_dataset
from basicgs.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicgs.pipelines import build_pipeline
from basicgs.viewers import build_viewer
from basicgs.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicgs.utils.misc import save_runtime_code
from basicgs.utils.options import copy_opt_file, dict2str, parse_options


def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join(opt['path']['experiments_root'], 'tb_logger'))
    return tb_logger


def is_time(current_iter, v):
    if isinstance(v, Number):
        return current_iter % v == 0
    elif isinstance(v, list):
        return current_iter in v
    else:
        raise NotImplementedError


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    val_sets = []
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio)
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            if dataset_opt.get('same_as_train', False):
                val_set = deepcopy(train_set)
            else:
                val_set = build_dataset(dataset_opt)
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_sets.append(val_set)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_set, val_sets, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('output', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set random seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join(opt['path']['experiments_root'], 'tb_logger'))

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])
    save_runtime_code(opt['path']['backup'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicgs', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, val_loaders, total_epochs, total_iters = result

    # create model
    model = build_pipeline(opt, train_loader)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    viewer = None
    if opt.get('viewer') is not None:
        viewer_opt = deepcopy(opt['viewer'])
        viewer_opt['exp_opt'] = opt
        viewer_opt['net_g'] = model.net_g
        viewer_opt['log_dir'] = opt['path']['experiments_root']
        viewer_opt['total_iters'] = total_iters
        viewer = build_viewer(viewer_opt)

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    # dataloader prefetcher
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        prefetcher.reset()
        train_data = prefetcher.next()

        while train_data is not None:
            data_timer.record()

            current_iter += 1
            if current_iter > total_iters:
                break
            # update learning rate
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()

            # log
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            # viewer
            if viewer is not None:
                with torch.no_grad():
                    try:
                        viewer.set_current_iter(current_iter, model.get_current_log())
                        viewer.update_gaussians(model.net_g)
                        viewer.update()
                    except Exception as e:
                        import traceback
                        viewer.logger.error(f'Error in viewer: {e}\n{traceback.format_exc()}')
                        viewer = None

            # save models and training states
            if is_time(current_iter, opt['logger']['save_checkpoint_freq']):
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)

            # validation
            if opt.get('val') is not None and is_time(current_iter, opt['val']['val_freq']):
                if len(val_loaders) > 1:
                    logger.warning('Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            data_timer.start()
            iter_timer.start()
            train_data = prefetcher.next()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
        model.validation(train_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
