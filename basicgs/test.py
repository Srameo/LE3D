import logging
import torch
from os import path as osp

from basicgs.data import build_dataset
from basicgs.pipelines import build_pipeline
from basicgs.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicgs.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicgs', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        if dataset_opt.get('same_as_train', False):
            logger.info(f'{_} is same as train, skip!')
            continue
        test_set = build_dataset(dataset_opt)
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_set)

    # create model
    model = build_pipeline(opt)

    for test_loader in test_loaders:
        test_set_name = test_loader.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
