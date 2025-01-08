import numpy as np
import tqdm
from basicgs.data.colmap_utils import qvec2rotmat, read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text
from basicgs.gaussians.util.graphics import focal2fov, fov2focal


NEAR_STRETCH = .9  # Push forward near bound for forward facing render path.
FAR_STRETCH = 5.  # Push back far bound for forward facing render path.
FOCUS_DISTANCE = .75  # Relative weighting of near, far bounds for render path.


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world

def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)

def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]

def recenter_poses(poses: np.ndarray):
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform

def generate_spiral_path(poses: np.ndarray,   # N, 4, 4
                         focal: np.ndarray,
                         n_frames: int = 120,
                         n_rots: int = 2,
                         zrate: float = .5,
                         focal_rate: float = 1.0,
                         radii_percent=90) -> np.ndarray:
    """Calculates a forward facing spiral path for rendering."""
    # Get radii for spiral path using 90th percentile of camera positions.
    poses, transform = recenter_poses(poses)
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), radii_percent, 0)
    radii = np.concatenate([radii, [1.]])


    # Generate poses for spiral path.
    render_poses = []
    cam2world = average_pose(poses)
    up = poses[:, :3, 1].mean(0)
    for theta in np.linspace(0., 2. * np.pi * n_rots, n_frames, endpoint=False):
      t = radii * [-np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]
      position = cam2world @ t
      lookat = cam2world @ [0, 0, -focal * focal_rate, 1.]
      z_axis = position - lookat
      render_poses.append(viewmatrix(z_axis, up, position))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = unpad_poses(np.linalg.inv(transform) @ pad_poses(render_poses))
    return render_poses


def generate_spiral_path_from_colmap(extrinsic_path, intrinsic_path,
                                     n_frames: int = 120,
                                     n_rots: int = 2,
                                     zrate: float = .5,
                                     focal_rate: float = 1.0,
                                     radii_percent=90) -> np.ndarray:
    cam_extrinsics = read_extrinsics_binary(extrinsic_path)
    cam_intrinsics = read_intrinsics_binary(intrinsic_path)
    assert len(cam_intrinsics) == 1
    poses = []
    count, focal_x, focal_y = 0, 0, 0
    for key in tqdm.tqdm(cam_extrinsics):
        count += 1
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        R = np.transpose(qvec2rotmat(extr.qvec))    # 3, 3
        T = np.array(extr.tvec)                     # 3
        pose = np.concatenate([R, T[..., None]], 1) # 3, 4
        poses.append(pose)
        height = intr.height
        width = intr.width

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            fov_y = focal2fov(focal_length_x, height)
            fov_x = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            fov_y = focal2fov(focal_length_y, height)
            fov_x = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        focal_x += focal_length_x
        focal_y += focal_length_y

    focal_x = focal_x / count
    focal_y = focal_y / count
    focal = focal_x
    print(focal_x, focal_length_y)
    poses = np.stack(poses, 0)

    return generate_spiral_path(poses, focal, n_frames, n_rots, zrate, focal_rate, radii_percent), \
           fov_x, fov_y
