import importlib
from copy import deepcopy
from os import path as osp

from basicgs.utils import get_root_logger, scandir
from basicgs.utils.registry import PIPE_REGISTRY

__all__ = ['build_pipeline']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_pipeline.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'basicgs.pipelines.{file_name}') for file_name in model_filenames]


def build_pipeline(opt, train_loader):
    """Build model from options.

    Args:
        opt (dict): Configuration. It must contain:
            pipe_type (str): Model type.
    """
    opt = deepcopy(opt)
    model = PIPE_REGISTRY.get(opt['pipe_type'])(opt, train_loader)
    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
