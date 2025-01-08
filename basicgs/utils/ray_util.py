import math
import torch
import numpy as np
from basicgs.gaussians.util.graphics import fov2focal

def rays_grid(cam, shape=None, *, to_batch=False):
    H, W = (cam.image_height, cam.image_width) if shape is None else shape
    device = cam.world_view_transform.device
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device),
        torch.linspace(0, H-1, H, device=device))
    i = i.t()
    j = j.t()

    focal_x = fov2focal(cam.fov_x, W)
    focal_y = fov2focal(cam.fov_y, H)
    c2w = cam.world_view_transform.transpose(0, 1).inverse()

    dirs = torch.stack([(i-W//2)/focal_x, -(j-H//2)/focal_y, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    rays_d = rays_d.permute(2, 0, 1)  # reshape to (C, H, W)
    rays_o = rays_o.permute(2, 0, 1)  # reshape to (C, H, W)
    if to_batch:
        return torch.cat([rays_o, rays_d], dim=0).unsqueeze(0)
    return rays_o, rays_d


def rays_grid_patchwise(cam, patch_num, shape=None, *, to_batch=False):
    H, W = (cam.image_height, cam.image_width) if shape is None else shape
    device = cam.world_view_transform.device
    patch_num_h, patch_num_w = patch_num
    # pytorch's meshgrid has indexing='ij'
    x, y = torch.linspace(0, W, patch_num_w+1, device=device), torch.linspace(0, H, patch_num_h+1, device=device)
    i, j = torch.meshgrid(
        (x[1:] + x[:-1]) / 2 - 1,
        (y[1:] + y[:-1]) / 2 - 1)
    i = i.t()
    j = j.t()

    focal_x = fov2focal(cam.fov_x, W)
    focal_y = fov2focal(cam.fov_y, H)
    c2w = cam.world_view_transform.transpose(0, 1).inverse()

    dirs = torch.stack([(i-W//2)/focal_x, -(j-H//2)/focal_y, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)

    rays_d = rays_d.permute(2, 0, 1)  # reshape to (C, H, W)
    rays_o = rays_o.permute(2, 0, 1)  # reshape to (C, H, W)
    if to_batch:
        return torch.cat([rays_o, rays_d], dim=0).unsqueeze(0)
    return rays_o, rays_d

def rays_cone_worldwise(ray_o, center, radius, normal, N, *, random_sample=True, to_batch=False):
    normal = normal / np.linalg.norm(normal)
    # 找到与法向量正交的向量
    if (normal == np.array([1, 0, 0])).all() or (normal == np.array([-1, 0, 0])).all():
        v = np.cross(normal, np.array([0, 1, 0]))
    else:
        v = np.cross(normal, np.array([1, 0, 0]))
    u = np.cross(normal, v)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)

    # 生成圆内的点
    if random_sample:
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, radius**2, N))
    else:
        angles = np.linspace(0, 2*np.pi, N)
        radii = np.sqrt(np.linspace(0, radius**2, N))
    points = np.array([np.cos(angles) * radii, np.sin(angles) * radii])

    # 将点从2D转换到3D
    points_3D = np.zeros((N, 3))
    points_3D[:,0] = center[0] + points[0,:]*u[0] + points[1,:]*v[0]
    points_3D[:,1] = center[1] + points[0,:]*u[1] + points[1,:]*v[1]
    points_3D[:,2] = center[2] + points[0,:]*u[2] + points[1,:]*v[2]

    ray_o = torch.tensor(ray_o).float().reshape(3, 1)
    rays_d = torch.tensor(points_3D).float().permute(1, 0) - ray_o # 3, N
    rays_o = ray_o.expand(rays_d.shape)  # 3, N
    if to_batch:
        return torch.cat([rays_o, rays_d], dim=0).unsqueeze(0)
    return rays_o, rays_d

def cone_grid_worldwise(origin, near, far, fov, normal, N, *, random_sample=True):
    def sample_from_0_to_1(size):
        if random_sample:
            return torch.rand(size)
        else:
            return torch.linspace(0, 1, size)

    depths = sample_from_0_to_1(N) ** (1 / 3.0) * (far - near) + near
    max_radius = depths * math.tan(fov / 2)
    radius = sample_from_0_to_1(N) ** (1 / 2.0) * max_radius
    thetas = sample_from_0_to_1(N) * 2 * torch.pi

    x = radius * torch.cos(thetas)
    y = radius * torch.sin(thetas)
    z = depths
    points = torch.vstack([x, y, z]).T

    # 检查normal是否与z轴平行
    z_axis = torch.tensor([0, 0, -1], dtype=torch.float32)
    normal = normal / torch.norm(normal)
    if torch.allclose(normal, z_axis) or torch.allclose(normal, -z_axis):
        # 如果normal指向z轴的负方向，则进行180度旋转
        if torch.allclose(normal, -z_axis):
            R = torch.diag(torch.tensor([1, 1, -1], dtype=torch.float32))
            points = torch.matmul(points, R.T)  # 应用旋转
    else:
        # 构造旋转矩阵，将z轴旋转到normal方向
        v = torch.linalg.cross(z_axis, normal)
        s = torch.norm(v)
        c = torch.dot(z_axis, normal)
        k = 1 - c
        Vx = torch.tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=torch.float32)
        R = torch.eye(3, dtype=torch.float32) + Vx + torch.matmul(Vx, Vx) * k / s**2
        points = torch.matmul(points, R.T)  # 应用旋转

    # 平移点到圆锥的顶点
    points += origin

    return points