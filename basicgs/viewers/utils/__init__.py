import numpy as np
import torch
import plotly.graph_objects as go
import scipy.interpolate as interp

def depth_naninf_to_red(depth_map):
    if depth_map.shape[0] == 1:
        depth_map = depth_map.repeat(3, 1, 1)
    depth_map = torch.nan_to_num(depth_map, 10.0)
    depth_map = depth_map.clip(0, 10.0)
    depth_map[depth_map > 1] = -1
    depth_map[0, depth_map[0] < 0] = 1
    depth_map[1, depth_map[0] < 0] = 0
    depth_map[2, depth_map[0] < 0] = 0
    return depth_map.clip(0, 1)

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

def get_wxyz(position, look_at, up):
    # 计算相机坐标系的三个轴
    forward = look_at - position
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)

    # 重新计算up以确保正交
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # 构建旋转矩阵
    R = np.array([
        [right[0], up[0], -forward[0]],
        [right[1], up[1], -forward[1]],
        [right[2], up[2], -forward[2]]
    ])

    # 从旋转矩阵计算四元数
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S

    return np.array([w, x, y, z])

def get_lut_fig_with_control_points(lut):
    x = np.array([0, 0.25, 0.5, 0.75, 1])
    y = lut
    spline = interp.UnivariateSpline(x, y, s=0, k=2)
    curve = spline(np.linspace(0, 1, 65536))
    curve = np.clip(curve, 0, 1).astype(np.float32)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0, 1, 65536), y=curve, mode='lines', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(color='red', size=8)))
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        template='plotly',
        showlegend=False
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def get_lut(control_points, bit_depth=16):
    x = np.array([0, 0.25, 0.5, 0.75, 1])
    y = control_points
    spline = interp.UnivariateSpline(x, y, s=0, k=2)
    curve = spline(np.linspace(0, 1, 2**bit_depth))
    curve = np.clip(curve, 0, 1).astype(np.float32)
    return torch.tensor(curve, device='cuda')