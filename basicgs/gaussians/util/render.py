import math
import torch
from torch.nn import functional as F
from basicgs.gaussians.gs_base_arch import GaussianBaseModel
from basicgs.gaussians.util.graphics import fov2focal
from kornia.filters import get_gaussian_kernel2d


from base_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettingsBase,
    GaussianRasterizer as GaussianRasterizerBase
)
from hist_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettings32,
    GaussianRasterizer as GaussianRasterizer32
)
from full_rasterization import (
    GaussianRasterizationSettings as GaussianRasterizationSettingsFull,
    GaussianRasterizer as GaussianRasterizerFull
)

EPS = 1.0e-3

def render(gs_model: GaussianBaseModel, viewpoint_camera, shape=None, *, bg_color_override=None):
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs_model.get_xyz, dtype=gs_model.get_xyz.dtype, requires_grad=True, device=gs_model.get_xyz.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

        raster_settings = GaussianRasterizationSettingsBase(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=gs_model.bg_color if bg_color_override is None else bg_color_override,
            scale_modifier=gs_model.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gs_model.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=gs_model.debug
        )

        rasterizer = GaussianRasterizerBase(raster_settings=raster_settings)

        means3D = gs_model.get_xyz
        means2D = screenspace_points
        opacity = gs_model.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if gs_model.compute_cov3D_python:
            cov3D_precomp = gs_model.get_covariance(gs_model.scaling_modifier)
        else:
            scales = gs_model.get_scaling
            rotations = gs_model.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if gs_model.convert_SHs_python:
            colors_precomp = gs_model.get_color(viewpoint_camera)
        else:
            shs = gs_model.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
        # rendered_image, radii, rendered_depth, rendered_final_opacity, gs_count_pack = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

def render_depth_raywise(gs_model: GaussianBaseModel, viewpoint_camera, shape=None, return_filter=False, return_all=False):
    # TODO: compute depth
    if shape is None:
        H, W = viewpoint_camera.image_height, viewpoint_camera.image_width
    else:
        H, W = shape
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    r = w2c[:3, :3]
    t = w2c[:3, -1:]
    point_at_cam_view = r @ gs_model.get_xyz.transpose(0, 1) + t
    point_at_cam_view = point_at_cam_view.transpose(0, 1)
    point_z = point_at_cam_view[:, -1:]

    focal_x = fov2focal(viewpoint_camera.fov_x, W)
    focal_y = fov2focal(viewpoint_camera.fov_y, H)
    c2i = torch.tensor([[focal_x, 0, W / 2],
                        [0, focal_y, H / 2],
                        [0, 0, 1]], dtype=torch.float32, device=point_at_cam_view.device)
    point_at_im_view = c2i @ point_at_cam_view.transpose(0, 1)
    point_at_im_view = point_at_im_view.transpose(0, 1)
    point_at_im_view = point_at_im_view / point_at_im_view[:, -1:]

    point_at_im_view = point_at_im_view.round().long()
    x, y, _ = point_at_im_view.transpose(0, 1)
    if (not return_all) or return_filter:
        pixel_filter_x = torch.logical_and(x >= 0, x < W)
        pixel_filter_y = torch.logical_and(y >= 0, y < H)
        pixel_filter = torch.logical_and(pixel_filter_x, pixel_filter_y)
    else:
        pixel_filter = None

    if return_all:
        return_pack = [x, y, point_z.squeeze()]
    else:
        xx = x[pixel_filter]
        yy = y[pixel_filter]
        zz = point_z.squeeze()[pixel_filter]
        return_pack = [xx, yy, zz]
    if return_filter:
        return_pack.append(pixel_filter)
    return return_pack


def render_depth(gs_model: GaussianBaseModel, viewpoint_camera, shape=None):
    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    r = w2c[:3, :3]
    t = w2c[:3, -1:]
    point_at_cam_view = r @ gs_model.get_xyz.transpose(0, 1) + t
    point_at_cam_view = point_at_cam_view.transpose(0, 1)
    point_z = point_at_cam_view[:, -1:]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs_model.get_xyz, dtype=gs_model.get_xyz.dtype, requires_grad=True, device=gs_model.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettingsFull(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(gs_model.bg_color), # TODO: replace bg_color as depth_max
        scale_modifier=gs_model.scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=gs_model.debug
    )

    rasterizer = GaussianRasterizerFull(raster_settings=raster_settings)

    means3D = gs_model.get_xyz
    means2D = screenspace_points
    opacity = gs_model.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if gs_model.compute_cov3D_python:
        cov3D_precomp = gs_model.get_covariance(gs_model.scaling_modifier)
    else:
        scales = gs_model.get_scaling
        rotations = gs_model.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # TODO: replace pre_computed_color as deptg
    colors_precomp = point_z.repeat(1, 3)
    colors_precomp = torch.max(torch.zeros_like(colors_precomp), colors_precomp)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    color, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "max_depth": point_z.max().item(),
            "radii": radii}


def render_full_package(gs_model: GaussianBaseModel, viewpoint_camera, shape=None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs_model.get_xyz, dtype=gs_model.get_xyz.dtype, requires_grad=True, device=gs_model.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettingsFull(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(gs_model.bg_color), # TODO: replace bg_color as depth_max
        scale_modifier=gs_model.scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=gs_model.debug
    )

    rasterizer = GaussianRasterizerFull(raster_settings=raster_settings)

    means3D = gs_model.get_xyz
    means2D = screenspace_points
    opacity = gs_model.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if gs_model.compute_cov3D_python:
        cov3D_precomp = gs_model.get_covariance(gs_model.scaling_modifier)
    else:
        scales = gs_model.get_scaling
        rotations = gs_model.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # TODO: replace pre_computed_color as deptg
    colors_precomp = gs_model.get_color(viewpoint_camera)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    color, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": color,
            "depth": depth,
            "median_depth": median_depth,
            "final_opacity": final_opacity,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def render_final_opacity(gs_model: GaussianBaseModel, viewpoint_camera, shape=None):
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs_model.get_xyz, dtype=gs_model.get_xyz.dtype, requires_grad=True, device=gs_model.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettingsBase(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(gs_model.bg_color), # TODO: replace bg_color as depth_max
        scale_modifier=gs_model.scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=gs_model.debug
    )

    rasterizer = GaussianRasterizerBase(raster_settings=raster_settings)

    means3D = gs_model.get_xyz
    means2D = screenspace_points
    opacity = gs_model.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if gs_model.compute_cov3D_python:
        cov3D_precomp = gs_model.get_covariance(gs_model.scaling_modifier)
    else:
        scales = gs_model.get_scaling
        rotations = gs_model.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # TODO: replace pre_computed_color as deptg
    colors_precomp = torch.ones((gs_model._xyz.shape[0], 3), device=gs_model._xyz.device)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_hist(gs_model: GaussianBaseModel, viewpoint_camera, num_bins=128, shape=None, near_far=None):

    NUM_CHANNELS = 32
    assert num_bins % NUM_CHANNELS == 0, 'num_bins must be divisible by NUM_CHANNELS!'

    w2c = viewpoint_camera.world_view_transform.transpose(0, 1)
    r = w2c[:3, :3]
    t = w2c[:3, -1:]
    point_at_cam_view = r @ gs_model.get_xyz.transpose(0, 1) + t
    point_at_cam_view = point_at_cam_view.transpose(0, 1)
    point_z = point_at_cam_view[:, -1]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(gs_model.get_xyz, dtype=gs_model.get_xyz.dtype, requires_grad=True, device=gs_model.get_xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettings32(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(NUM_CHANNELS, device=gs_model._xyz.device), # TODO: replace bg_color as depth_max
        scale_modifier=gs_model.scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False
    )

    rasterizer = GaussianRasterizer32(raster_settings=raster_settings)

    means3D = gs_model.get_xyz
    means2D = screenspace_points
    opacity = gs_model.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if gs_model.compute_cov3D_python:
        cov3D_precomp = gs_model.get_covariance(gs_model.scaling_modifier)
    else:
        scales = gs_model.get_scaling
        rotations = gs_model.get_rotation


    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    # TODO: replace pre_computed_color as depth
    with torch.no_grad():
        # near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        # near, far = point_z.min().item(), point_z.max().item()
        near, far = (max(point_z.min().item(), 0), point_z.max().item()) if near_far is None else near_far
        z_blocksize = (far - near) / (num_bins - 1)
        point_z_block = (point_z - near) / z_blocksize
        point_z_block = torch.floor(point_z_block).long().clip(0, num_bins-1)
        colors_precomp = torch.nn.functional.one_hot(point_z_block, num_classes=num_bins).float()
        if near_far is None:
            zero_filter = torch.logical_or(point_z < near, point_z > far)
            colors_precomp[zero_filter, :] = 0

    if num_bins == NUM_CHANNELS:
        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "near": near, "far": far,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

    else:
        rendered_hist = []
        ## split forward
        forward_time = num_bins // NUM_CHANNELS
        for curr_forward in range(forward_time):
            curr_colors_precomp = colors_precomp[:, curr_forward*NUM_CHANNELS:(curr_forward+1)*NUM_CHANNELS]
            rendered_image, radii, _ = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = shs,
                colors_precomp = curr_colors_precomp,
                opacities = opacity,
                scales = scales,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)
            rendered_hist.append(rendered_image)
        return {"render": torch.cat(rendered_hist),
                "near": near, "far": far,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii}

def render_depth_with_filter(gs_model: GaussianBaseModel, viewpoint_camera, mask, shape=None):
    xyz = gs_model.get_xyz[mask, :]
    opacity = gs_model.get_opacity[mask, :]
    scales = gs_model.get_scaling[mask, :]
    rotations = gs_model.get_rotation[mask, :]

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=xyz.dtype, requires_grad=True, device=xyz.device) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
    tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

    H, W = shape if shape is not None else (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))

    raster_settings = GaussianRasterizationSettingsFull(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros_like(gs_model.bg_color), # TODO: replace bg_color as depth_max
        scale_modifier=gs_model.scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gs_model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=gs_model.debug
    )

    rasterizer = GaussianRasterizerFull(raster_settings=raster_settings)

    means3D = xyz
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = torch.zeros((xyz.shape[0], 3), device=xyz.device)

    _, radii, depth, median_depth, final_opacity = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": depth,
            "final_opacity": final_opacity,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_near_far(gs_model: GaussianBaseModel, viewpoint_camera, near_far_indexes, shape=None):
    with torch.no_grad():
        near_indexes = near_far_indexes[0]
        far_indexes = near_far_indexes[1]
        uniq_near_indexes = torch.unique(near_indexes[near_indexes != -1]).long()
        uniq_far_indexes = torch.unique(far_indexes[far_indexes != -1]).long()
        near_masks = torch.zeros_like(gs_model._opacity).squeeze()
        near_masks[uniq_near_indexes] = 1
        far_masks  = torch.zeros_like(gs_model._opacity).squeeze()
        far_masks[uniq_far_indexes] = 1
        near_masks = near_masks.bool()
        far_masks  = far_masks.bool()

    far_pack  = render_depth_with_filter(gs_model, viewpoint_camera, far_masks,  shape=shape)
    near_pack = render_depth_with_filter(gs_model, viewpoint_camera, near_masks, shape=shape)

    return {'near': near_pack['render'],
            'near_final_opacity': near_pack['final_opacity'],
            'far': far_pack['render'],
            'far_final_opacity': far_pack['final_opacity']}
