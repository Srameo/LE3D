import math
from basicgs.utils.logger import get_root_logger
import torch
from torch import nn
import numpy as np
from plyfile import PlyData, PlyElement
from copy import deepcopy

from .util import inverse_sigmoid, build_scaling_rotation, strip_symmetric, build_rotation
from .util import rgb2sh, eval_sh
from .util import BasicPointCloud
from .util import fetch_ply
from torch import nn
from simple_knn._C import distCUDA2

from gcnt_rasterization import GaussianRasterizationSettings, GaussianRasterizer, TOP_K_NEAR_OR_FAR
from basicgs.utils.registry import ARCH_REGISTRY

@ARCH_REGISTRY.register()
class GaussianBaseModel(nn.Module):
    default_viewer_type = 'BaseViewer'

    def __init__(self, init_ply_path, sh_degree : int,
                 sh_degree_up_per_iter, percent_dense,
                 densify_from_iter, densify_until_iter,
                 densification_interval, densify_grad_threshold,
                 opacity_reset_interval, prune_opacity_threshold,
                 reset_opacity_value_max=0.01,
                 scaling_modifier=1.0, bg_color=[0, 0, 0], center_color=0.5,
                 compute_cov3D_python=False, convert_SHs_python=False,
                 densify_size_threshold=20,
                 debug=False):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.center_color = center_color
        self.setup_functions()
        self.init_ply_path = init_ply_path
        self.scaling_modifier = scaling_modifier
        self.debug = debug
        self.percent_dense = percent_dense
        self.reset_opacity_value_max = reset_opacity_value_max

        self.sh_degree_up_per_iter = sh_degree_up_per_iter
        self.densify_until_iter = densify_until_iter
        self.densify_from_iter = densify_from_iter
        self.densification_interval = densification_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_size_threshold = densify_size_threshold
        self.opacity_reset_interval = opacity_reset_interval
        self.prune_opacity_threshold = prune_opacity_threshold
        self.white_background = bg_color == [1, 1, 1]

        self.bg_color = nn.Parameter(torch.tensor(bg_color).float(), requires_grad=False)
        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

        self.create_from_pcd(fetch_ply(self.init_ply_path))

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, device=self._xyz.device)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_color(self, viewpoint_camera):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors = sh2rgb + self.center_color
        colors_precomp = torch.clamp_min(colors, 0.0)
        return colors_precomp

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*self.reset_opacity_value_max))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def sh_degree_up(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = rgb2sh(torch.tensor(np.asarray(pcd.colors)).float(), self.center_color)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialisation : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.denom = torch.zeros((self.get_xyz.shape[0], 1)).cuda()

    def save_ply(self, path):
        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            # All channels except the 3 DC
            for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
                l.append('f_dc_{}'.format(i))
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
            l.append('opacity')
            for i in range(self._scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_features(self, extra_f_names, plydata, point_count):
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((point_count, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        return features_extra

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        features_extra = self.load_features(extra_f_names, plydata, xyz.shape[0])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def fetch_parameters_for_optimizer(self):
        return {
            'position': {'params': [self._xyz], 'name': 'xyz'},
            'feature_dc':  {'params': [self._features_dc], 'name': 'f_dc'},
            'feature_rest': {'params': [self._features_rest], 'name': 'f_rest'},
            'scaling': {'params': [self._scaling], 'name': 'scaling'},
            'rotation': {'params': [self._rotation], 'name': 'rotation'},
            'opacity': {'params': [self._opacity], 'name': 'opacity'}
        }

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def finalize_iter(self, current_iter, render_pkg, cameras_extent, optimizer, *args):
        if current_iter % self.sh_degree_up_per_iter == 0:
            self.sh_degree_up()
        if current_iter < self.densify_until_iter:
            self.optimizer = optimizer
            visibility_filter = render_pkg['visibility_filter']
            viewspace_point_tensor = render_pkg['viewspace_points']
            radii = render_pkg["radii"]
            self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
            self.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if current_iter > self.densify_from_iter and current_iter % self.densification_interval == 0:
                size_threshold = self.densify_size_threshold if current_iter > self.opacity_reset_interval else None
                self.densify_and_prune(self.densify_grad_threshold, self.prune_opacity_threshold, cameras_extent, size_threshold)

            if current_iter % self.opacity_reset_interval == 0 or (self.white_background and current_iter == self.densify_from_iter):
                self.reset_opacity()

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3), device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask], device=self._xyz.device).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self._xyz.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        # new_opacities = self.inverse_opacity_activation(1 - torch.sqrt(1 - self.opacity_activation(new_opacities)))
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        logger = get_root_logger()
        before_densify_prune_mask = (self.denom == 0).squeeze()
        self.prune_points(before_densify_prune_mask)

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        logger.info(f"gs_out_of_range = {before_densify_prune_mask.sum()}")

        def log_gaussian_num(text=''):
            cnt1 = (self.get_opacity < self.prune_opacity_threshold).sum()
            cnt2 = torch.logical_and(self.prune_opacity_threshold <= self.get_opacity,
                                     self.get_opacity <= self.reset_opacity_value_max).sum()
            cnt3 = (self.reset_opacity_value_max < self.get_opacity).sum()
            logger.info(f"{int(cnt1)}+{int(cnt2)}+{int(cnt3)}={int(self._opacity.shape[0])} Gaussians {text}...")

        log_gaussian_num('before densify_and_clone()')
        self.densify_and_clone(grads, max_grad, extent)
        log_gaussian_num(f'after densify_and_clone(max_grad={max_grad}, extent={extent})')
        self.densify_and_split(grads, max_grad, extent)
        log_gaussian_num(f'after densify_and_split(max_grad={max_grad}, extent={extent})')

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            logger.info(f"big_points_vs = {int(big_points_vs.sum())}, big_points_ws = {big_points_ws.sum()}, extent={extent}")
        self.prune_points(prune_mask)
        log_min_op = min_opacity.min().item() if torch.is_tensor(min_opacity) else min_opacity
        log_gaussian_num(f'after prune_points(min_opacity={log_min_op:.4f}, max_screen_size={max_screen_size})')

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def load_state_dict(self, state_dict, strict=True):
        xyz = state_dict.pop('_xyz').to(self._xyz.device)
        features_dc = state_dict.pop('_features_dc').to(self._xyz.device)
        features_extra = state_dict.pop('_features_rest').to(self._xyz.device)
        opacities = state_dict.pop('_opacity').to(self._xyz.device)
        scales = state_dict.pop('_scaling').to(self._xyz.device)
        rots = state_dict.pop('_rotation').to(self._xyz.device)
        bg_color = state_dict.pop('bg_color').to(self._xyz.device)

        self._xyz = nn.Parameter(xyz, True)
        self._features_dc = nn.Parameter(features_dc, True)
        self._features_rest = nn.Parameter(features_extra, True)
        self._opacity = nn.Parameter(opacities, True)
        self._scaling = nn.Parameter(scales, True)
        self._rotation = nn.Parameter(rots, True)
        self.bg_color = nn.Parameter(bg_color, False)
        self.active_sh_degree = self.max_sh_degree

        if strict and len(state_dict) > 0:
            raise ValueError('Unrecognized state_dict keys: {}'.format(state_dict.keys()))

    def forward(self, viewpoint_camera):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.get_xyz, dtype=self.get_xyz.dtype, requires_grad=True, device=self.get_xyz.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.get_xyz
        means2D = screenspace_points
        opacity = self.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if self.compute_cov3D_python:
            cov3D_precomp = self.get_covariance(self.scaling_modifier)
        else:
            scales = self.get_scaling
            rotations = self.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.convert_SHs_python:
            colors_precomp = self.get_color(viewpoint_camera)
        else:
            shs = self.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, gs_count_pack, radii = rasterizer(
        # rendered_image, radii, rendered_depth, rendered_final_opacity, gs_count_pack = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        gs_count, near_indexes, far_indexes = gs_count_pack[:1], gs_count_pack[1:1+TOP_K_NEAR_OR_FAR], gs_count_pack[1+TOP_K_NEAR_OR_FAR:1+2*TOP_K_NEAR_OR_FAR]

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                # "depth": rendered_depth,
                # "final_opacity": rendered_final_opacity,
                "gs_count": gs_count.long(),
                "near_indexes": near_indexes.long(),
                "far_indexes": far_indexes.long(),
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

class PcdBaseModel(nn.Module):
    def __init__(self, gs_model: GaussianBaseModel) -> None:
        super().__init__()
        self.max_sh_degree = gs_model.max_sh_degree
        self.active_sh_degree = gs_model.active_sh_degree
        self.bg_color = deepcopy(gs_model.bg_color)
        self.scaling_modifier = gs_model.scaling_modifier
        self.debug = gs_model.debug
        self.compute_cov3D_python = gs_model.compute_cov3D_python
        self.convert_SHs_python = gs_model.convert_SHs_python
        self.setup_functions()
        self.create_from_pcd(fetch_ply(gs_model.init_ply_path))

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation, device=self._xyz.device)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float()
        fused_color = rgb2sh(torch.tensor(np.asarray(pcd.colors)).float())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialisation for pcd base model: {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2 / fused_point_cloud.shape[0]))[...,None].repeat(1, 3)
        # scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        opacities = inverse_sigmoid(1*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(False), False)
        self._rotation = nn.Parameter(rots.requires_grad_(False), False)
        self._opacity = nn.Parameter(opacities.requires_grad_(False), False)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)