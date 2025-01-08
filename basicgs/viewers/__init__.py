import importlib
from copy import deepcopy
from os import path as osp

from basicgs.utils import get_root_logger, scandir
from basicgs.utils.registry import VIEWER_REGISTRY

from .editor import Editor

__all__ = ['build_viewer', 'Editor']

# automatically scan and import viewer modules for registry
# scan all the files under the 'viewers' folder and collect files ending with '_viewer.py'
viewer_folder = osp.dirname(osp.abspath(__file__))
viewer_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(viewer_folder) if v.endswith('_viewer.py')]
# import all the viewer modules
_viewer_modules = [importlib.import_module(f'basicgs.viewers.{file_name}') for file_name in viewer_filenames]


def build_viewer(opt):
    opt = deepcopy(opt)
    viewer_type = opt.pop('type')
    viewer = VIEWER_REGISTRY.get(viewer_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Viewer [{viewer.__class__.__name__}] is created.')
    return viewer
