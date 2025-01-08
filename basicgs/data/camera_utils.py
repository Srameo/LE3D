import copy
from numpy import array
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from basicgs.gaussians.util import get_world_to_view_wts, get_proj_matrix
from basicgs.gaussians.util.graphics import fov2focal

class Camera(nn.Module):
    @torch.no_grad()
    def __init__(self, colmap_id, R, T, fov_x, fov_y, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 znear=0.01, zfar=100.0,
                 meta_data=None,
                 **kwargs
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.image_name = image_name

        self.original_image = image
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.focal_y = fov2focal(self.fov_y, self.image_height)
        self.focal_x = fov2focal(self.fov_x, self.image_width)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale
        self.meta_data = meta_data

        self.world_view_transform = torch.tensor(get_world_to_view_wts(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = get_proj_matrix(znear=self.znear, zfar=self.zfar, fov_x=self.fov_x, fov_y=self.fov_y).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.original_image = nn.Parameter(self.original_image, False)
        self.full_proj_transform = nn.Parameter(self.full_proj_transform, False)
        self.world_view_transform = nn.Parameter(self.world_view_transform, False)
        self.projection_matrix = nn.Parameter(self.projection_matrix, False)
        self.camera_center = nn.Parameter(self.camera_center, False)
        if self.meta_data is not None:
            self.meta_data = list(self.meta_data)[:-1]
            for d in self.meta_data:
                if hasattr(d, 'requires_grad'):
                    d.requires_grad = False
            self.meta_data = nn.ParameterList(self.meta_data)

@torch.no_grad()
def resize_camera(cam: Camera, scale_factor, mode='bicubic'):
    new_cam = copy.deepcopy(cam)
    width, height = cam.image_width, cam.image_height
    new_width, new_height = int(width * scale_factor), int(height * scale_factor)
    if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        new_img = F.interpolate(cam.original_image.unsqueeze(0), size=(new_height, new_width), mode=mode, align_corners=False).squeeze(0)
    else:
        new_img = F.interpolate(cam.original_image.unsqueeze(0), size=(new_height, new_width), mode=mode).squeeze(0)
    new_cam.original_image = nn.Parameter(new_img, False)

    new_cam.focal_y = fov2focal(cam.fov_y, new_height)
    new_cam.focal_x = fov2focal(cam.fov_x, new_width)

    return new_cam