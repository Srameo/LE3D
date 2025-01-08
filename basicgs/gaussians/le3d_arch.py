from copy import deepcopy

import numpy as np
import torch
from torch import nn
from simple_knn._C import distCUDA2

from basicgs.gaussians.gs_base_arch import GaussianBaseModel
from basicgs.gaussians.util.graphics import BasicPointCloud
from basicgs.gaussians.util.op import inverse_sigmoid
from basicgs.utils.logger import get_root_logger
from basicgs.utils.registry import ARCH_REGISTRY
from basicgs.utils.ray_util import rays_cone_worldwise, cone_grid_worldwise
from basicgs.gaussians.util.render import render_depth_raywise, render_hist

MIN_COLOR_DC_HARD = torch.tensor(-5.0)


@torch.no_grad()
def calculate_init_color_for_each_point(init_points, train_dataset, max_shutter=None, unique=True):

    class InitPoints:
        def __init__(self, points) -> None:
            self.points = torch.tensor(points)
        @property
        def get_xyz(self):
            return self.points

    def get_indices_of_min_z(x, y, z):
        # Step 1: Combine x and y into a 2D tensor
        combined = torch.stack((x, y), dim=1)

        # Step 2: Get unique rows and their corresponding indices
        unique_combined, inverse_indices = torch.unique(combined, return_inverse=True, dim=0)

        # Step 3: Find the index of the minimal z for each unique pair
        min_z_indices = torch.zeros(len(unique_combined), dtype=torch.long)

        # Initialize the minimal z values with a large number or infinity
        min_z_values = torch.full((len(unique_combined),), float('inf'))

        # Loop through all indices and update the minimal z values and their indices
        for index in range(len(z)):
            group_index = inverse_indices[index]
            if z[index] < min_z_values[group_index]:
                min_z_values[group_index] = z[index]
                min_z_indices[group_index] = index

        return min_z_indices

    def get_indexes(cam, points):
        x, y, z, pixel_filter = render_depth_raywise(points, cam, return_filter=True)
        if unique:
            first_occurrence_indices = get_indices_of_min_z(x, y, z)
            x = x[first_occurrence_indices]
            y = y[first_occurrence_indices]
            return x.long(), y.long(), first_occurrence_indices
        else:
            return x.long(), y.long(), pixel_filter

    init_points = InitPoints(init_points)
    color_count = torch.zeros(init_points.points.shape[0], 1)
    color_sum   = torch.zeros(init_points.points.shape[0], 3)

    for data in train_dataset:
        cam = data['camera']
        x, y, visible_filter = get_indexes(cam, init_points)
        im = cam.original_image
        if max_shutter is not None:
            im = im / cam.meta_data[4] * max_shutter
        colors = im[:, y, x].permute(1, 0)
        color_count[visible_filter] += 1
        color_sum[visible_filter] += colors

    init_colors = color_sum / color_count
    init_colors[torch.isinf(init_colors)] = 1.0e-8
    init_colors[torch.isnan(init_colors)] = 1.0e-8
    return init_colors.numpy()


@ARCH_REGISTRY.register()
class LE3DModel(GaussianBaseModel):
    default_viewer_type = 'Le3dViewer'

    def __init__(self, init_ply_path,
                 train_dataset, color_mlp_opt, color_feat_opt,
                 multi_exposure_training, add_init_points,
                 percent_dense, densify_from_iter, densify_until_iter,
                 densification_interval, densify_grad_threshold,
                 opacity_reset_interval, prune_opacity_threshold,
                 reset_opacity_value_max=0.01,
                 scaling_modifier=1.0, bg_color=[0, 0, 0], center_color=0.5,
                 compute_cov3D_python=False, convert_SHs_python=False,
                 densify_size_threshold=20,
                 debug=False):
        self.train_dataset = train_dataset
        self.color_mlp_opt = color_mlp_opt
        self.color_feat_opt = color_feat_opt
        self.multi_exposure_training = multi_exposure_training
        self.add_init_points = add_init_points
        super().__init__(init_ply_path,
                        3, 1500, # sh params, not working
                        percent_dense, densify_from_iter, densify_until_iter,
                        densification_interval, densify_grad_threshold,
                        opacity_reset_interval, prune_opacity_threshold,
                        reset_opacity_value_max, scaling_modifier, bg_color,
                        center_color, compute_cov3D_python, convert_SHs_python,
                        densify_size_threshold, debug)

        # init color mlp
        color_mlp_opt = deepcopy(color_mlp_opt)
        network_type = color_mlp_opt.pop('type')
        net = ARCH_REGISTRY.get(network_type)(**color_mlp_opt)
        logger = get_root_logger()
        logger.info(f'ColorMLP [{net.__class__.__name__}] is created.')
        self.color_mlp = net
        if hasattr(self, 'train_dataset') and self.train_dataset is not None:
            with torch.no_grad():
                # random sample a camera to initialize color mlp
                cam_id = torch.randint(0, len(self.train_dataset), (1,)).item()
                viewpoint_camera = self.train_dataset[cam_id]['camera']
                dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_xyz.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                colors_precomp = torch.log(self.color_mlp(self.get_features, dir_pp_normalized, w_bias=False))
                self._features_dc = nn.Parameter(self._features_dc - colors_precomp[:, None, :], requires_grad=True)

            # init exposure parameters
            if self.multi_exposure_training:   # not for inference
                assert hasattr(self.train_dataset, 'shutter_set')
                max_shutter, shutter_set = self.train_dataset.shutter_set
                self.register_exposure_parameters(shutter_set, max_shutter)

    # for multi exposure training
    def get_shutter_key(self, shutter):
        shutter = str(shutter).replace('.', '_')  # key must be str, and without .
        return shutter

    @property
    def get_features(self):
        return self._features_dc[:, 0, :], self._features_rest[:, 0, :]

    def get_color(self, viewpoint_camera):
        dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.get_xyz.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = self.color_mlp(self.get_features, dir_pp_normalized)
        return colors_precomp

    def register_exposure_parameters(self, shutter_set, max_shutter):
        self.max_shutter = max_shutter.item()
        exposure_correction_params = \
            nn.ParameterDict({
                ## must convert into str, and no .
                self.get_shutter_key(shutter): nn.Parameter(
                    torch.ones(3, 1, 1),
                    requires_grad=self.get_shutter_key(self.max_shutter)!=self.get_shutter_key(shutter)
                )
                for shutter in shutter_set
            })
        self.exposure_correction_params = exposure_correction_params

    def get_exposure_correction_param(self, shutter):
        shutter = self.get_shutter_key(shutter)
        return self.exposure_correction_params[shutter]

    def save_ply(self, path):
        super().save_ply(path)
        torch.save(self.color_mlp.state_dict(),
                    path.replace('.ply', '.pth').replace('net_g', 'color_mlp'))
        if hasattr(self, 'exposure_correction_params'):
            torch.save(self.exposure_correction_params,
                       path.replace('.ply', '.pth').replace('net_g', 'exposure_correction_params'))

    def load_features(self, extra_f_names, plydata, point_count):
        features_extra = np.zeros((point_count, len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], self.color_feat_opt['feat_len'], 1))
        return features_extra

    def load_ply(self, path):
        super().load_ply(path)
        self.color_mlp.load_state_dict(torch.load(path.replace('.ply', '.pth').replace('net_g', 'color_mlp')))

    @torch.no_grad()
    def create_from_pcd(self, pcd : BasicPointCloud):
        new_color = 1.0e-9
        if self.add_init_points is not None and hasattr(self, 'train_dataset') and self.train_dataset is not None:
            if hasattr(self, 'train_dataset') and self.train_dataset is not None:
                ims = []
                for data in self.train_dataset:
                    cam = data['camera']
                    im = cam.original_image
                    if hasattr(self, 'max_shutter'):
                        im = im / cam.meta_data[4] * self.max_shutter
                    ims.append(im)
                ims = torch.stack(ims, dim=0) # B, 3, H, W
                new_color = ims.mean(dim=(0, 2, 3))[None, :].numpy()

            ########## cone_ray
            N = self.add_init_points['N']
            radius_decay = self.add_init_points['radius_decay']
            fov_decay = self.add_init_points.get('fov_decay', 1.0)
            far_multiplier = self.add_init_points['far_multiplier']
            near_multiplier = self.add_init_points['near_multiplier']
            add_point_multiplier = self.add_init_points['add_point_multiplier']
            random_sample = self.add_init_points['random_sample']

            cams = [data['camera'] for data in self.train_dataset]
            cam_normal_vectors = []
            cam_centers = []
            cam_fovs = []
            for cam in cams:
                cam_w2c = cam.world_view_transform.transpose(0, 1)
                cam_normal_vec = -cam_w2c[2, :3]
                cam_normal_vectors.append(cam_normal_vec.numpy())
                cam_centers.append(cam.camera_center.numpy())
                cam_fovs.append(max(cam.fov_x, cam.fov_y))

            cam_normal_vectors = np.stack(cam_normal_vectors, axis=0)
            plane_normal_vec = cam_normal_vectors.mean(axis=0)
            plane_normal_vec = plane_normal_vec / np.linalg.norm(plane_normal_vec)
            cam_fov = max(cam_fovs) * fov_decay
            cam_centers = np.stack(cam_centers, axis=0)
            cam_center = cam_centers.mean(axis=0) # 3
            max_dist = np.linalg.norm(cam_centers - cam_center, axis=1).max() * radius_decay
            new_focal  = max_dist / np.tan(cam_fov / 2)
            new_center = cam_center - new_focal * plane_normal_vec

            dist = np.linalg.norm(pcd.points - new_center, axis=1)
            near, far = dist.min(), dist.max()
            add_point_count = add_point_multiplier * pcd.points.shape[0]
            if N is not None:
                rays_o, rays_d = rays_cone_worldwise(new_center, cam_center, max_dist, plane_normal_vec, N,
                                                    random_sample=random_sample)
                rays_o = rays_o.numpy().reshape(3, -1).T # B, 3
                rays_d = rays_d.numpy().reshape(3, -1).T # B, 3
                rays_d = rays_d / np.linalg.norm(rays_d, axis=1, keepdims=True) # to unit vec
                add_point_per_ray = int(add_point_count / N)
                if not random_sample:
                    ray_multiplier = -np.linspace(near * near_multiplier, far * far_multiplier, add_point_per_ray)  # (N)
                    new_points = rays_d.reshape(-1,1,3) * ray_multiplier.reshape(1,-1,1) + rays_o.reshape(-1, 1, 3)
                else:
                    ray_multiplier = -np.random.uniform(
                        near * near_multiplier, far * far_multiplier, (N, add_point_per_ray))  # (N, x)
                    new_points = rays_d.reshape(-1,1,3) * ray_multiplier[..., None] + rays_o.reshape(-1, 1, 3)
            else:
                new_points = cone_grid_worldwise(
                    torch.tensor(new_center), near * near_multiplier, far * far_multiplier, cam_fov,
                    torch.tensor(plane_normal_vec), add_point_count, random_sample=random_sample).numpy()
            new_points = new_points.reshape(-1, 3)
            new_colors = np.ones((new_points.shape[0], 3)) * new_color
            init_points = np.concatenate((pcd.points, new_points), axis=0)
            init_colors = np.concatenate((pcd.colors, new_colors), axis=0)

            new_points_count = new_points.shape[0]
            new_points_opacity = self.add_init_points.get('init_opacity', 0.1)

        else:
            init_points, init_colors = pcd.points, pcd.colors
            new_points_count = 0
            new_points_opacity = None

        if self.color_feat_opt.get('recalculate_color_by_pcd') and hasattr(self, 'train_dataset') and self.train_dataset is not None:
            max_shutter = getattr(self, 'max_shutter', None)
            if new_points_count > 0:
                original_points = pcd.points
                recalculated_init_colors = calculate_init_color_for_each_point(original_points, self.train_dataset, max_shutter=max_shutter)
                init_colors[:original_points.shape[0]] = recalculated_init_colors
                init_colors[original_points.shape[0]:] = calculate_init_color_for_each_point(init_points[original_points.shape[0]:], self.train_dataset, max_shutter, unique=False)
            else:
                init_colors = calculate_init_color_for_each_point(init_points, self.train_dataset, max_shutter)

        fused_point_cloud = torch.tensor(np.asarray(init_points)).float()
        inv_acted_color = np.log(np.asarray(init_colors))
        inv_acted_color[np.isnan(inv_acted_color)] = MIN_COLOR_DC_HARD
        fused_color = torch.tensor(inv_acted_color).float()
        features = torch.zeros((fused_color.shape[0], 3, 1)).float()
        features[:, :3, 0 ] = fused_color
        min_color_dc = self.color_feat_opt.get('min_color_dc', MIN_COLOR_DC_HARD)
        if min_color_dc == 'adaptive':
            feats = features[:-new_points_count] if new_points_count > 0 else features
            feats = feats[torch.logical_not(torch.isinf(feats))]
            min_color_dc = torch.max(MIN_COLOR_DC_HARD, torch.min(feats))
        features_dc = features[:,:,0:1].clip(min_color_dc, None)
        self._features_dc = nn.Parameter(features_dc.transpose(1, 2).contiguous().requires_grad_(True))
        # using color_mlp
        color_feat_length = self.color_feat_opt['feat_len']
        color_feat_sigma =  self.color_feat_opt['feat_init_sigma']
        self._features_rest = nn.Parameter(
            torch.randn((fused_point_cloud.shape[0], 1, color_feat_length)) * color_feat_sigma, requires_grad=True)

        print(f"Number of points at initialisation : {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(init_points)).float().cuda()), 0.0000001).cpu()
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4))
        rots[:, 0] = 1

        if new_points_count == 0:
            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float))
        else:
            opacities_original = inverse_sigmoid(0.1 * torch.ones((pcd.points.shape[0], 1), dtype=torch.float))
            opacities_new = inverse_sigmoid(
                new_points_opacity * torch.ones((new_points_count, 1), dtype=torch.float))
            opacities = torch.cat([opacities_original, opacities_new], 0)

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0])).cuda()
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1)).cuda()
        self.denom = torch.zeros((self.get_xyz.shape[0], 1)).cuda()

    def render_hist(self, viewpoint_camera, num_bins=32, shape=None, near_far=None):
        return render_hist(self, viewpoint_camera, num_bins, shape, near_far)