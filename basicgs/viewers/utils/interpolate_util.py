from copy import deepcopy
import torch
from viser import GuiApi
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import scipy.interpolate as interp
import numpy as np
from basicgs.viewers.utils import get_lut_fig_with_control_points
from basicgs.viewers.utils.frame_util import Frame, KeyFrame

def interpolate_poses(pose1, pose2, t):
    # 提取平移部分
    T1 = pose1[:3, 3]
    T2 = pose2[:3, 3]

    # 提取旋转部分并转换为四元数
    R1 = R.from_matrix(pose1[:3, :3])
    R2 = R.from_matrix(pose2[:3, :3])

    # 平移部分使用线性插值
    interpolated_T = (1 - t) * T1 + t * T2

    # 旋转部分使用球面线性插值
    slerp = Slerp([0, 1], R.from_quat([R1.as_quat(), R2.as_quat()]))
    interpolated_R = slerp([t]).as_matrix()[0]

    # 重新组合成4x4矩阵
    interpolated_pose = np.eye(4)
    interpolated_pose[:3, :3] = interpolated_R
    interpolated_pose[:3, 3] = interpolated_T

    return interpolated_pose

def interpolate(x1, x2, t):
    # 线性插值
    interpolated_wb = (1 - t) * x1 + t * x2
    return interpolated_wb

def interpolate_poses_with_acceleration(pose1, pose2, k, control_points):
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) * (k-1)
    y = np.array(control_points) * (k-1)
    spline = interp.UnivariateSpline(x, y, s=0, k=2)
    curve = spline(np.arange(k - 1))
    curve = (curve / (k-1)).clip(0, 1)

    interpolated_poses = [pose1]
    for t in curve:
        interpolated_pose = interpolate_poses(pose1, pose2, t)
        interpolated_poses.append(interpolated_pose)
    interpolated_poses.append(pose2)

    return interpolated_poses

class KeyInterpolation:
    def __init__(self, editor, order, gui: GuiApi, fps: float, keyframe1: KeyFrame, keyframe2: KeyFrame, interval: float=0.5, interval_for_blending: float = 0.1):
        self.editor = editor
        self.gui = gui
        self.interpolation = Interpolation(interval, fps, keyframe1, keyframe2, interval_for_blending)
        self._order = order
        self._folder = self.gui.add_folder(f'Interpolation {order}', order=order*2+101, expand_by_default=False)
        with self._folder:
            self.interval_text = self.gui.add_text(f'Interval (s):', str(interval))
            @self.interval_text.on_update
            def _(_):
                self.interpolation.interval = float(self.interval_text.value)

            self.interval_for_blending_text = self.gui.add_text(f'Interval for blending (s):', str(interval_for_blending))
            @self.interval_for_blending_text.on_update
            def _(_):
                self.interpolation.interval_for_blending = float(self.interval_for_blending_text.value)

            self.view_button = self.gui.add_button('View', hint='View this interpolation')
            @self.view_button.on_click
            def _(_):
                self.editor.preview_camera = None
                self.editor.preview_interpolation = self.order
                self.editor.logger.info(f'Preview interpolation {self._order}')

            with self.gui.add_folder('Acceleration'):
                self.apply_acceleration_checkbox = self.gui.add_checkbox('Apply acceleration', True)
                self.acceleration_fig_plotly = self.gui.add_plotly(get_lut_fig_with_control_points([0.0, 0.1, 0.5, 0.9, 1.0]), aspect=1.0, visible=self.apply_acceleration_checkbox.value)
                init_acceleration_25, init_acceleration_50, init_acceleration_75 = 0.1, 0.5, 0.9
                self.acceleration_25_control_point_slider = self.gui.add_slider('Control Point (0.25)', min=0, max=1, step=0.01, initial_value=init_acceleration_25, visible=self.apply_acceleration_checkbox.value)
                self.acceleration_50_control_point_slider = self.gui.add_slider('Control Point (0.50)', min=0, max=1, step=0.01, initial_value=init_acceleration_50, visible=self.apply_acceleration_checkbox.value)
                self.acceleration_75_control_point_slider = self.gui.add_slider('Control Point (0.75)', min=0, max=1, step=0.01, initial_value=init_acceleration_75, visible=self.apply_acceleration_checkbox.value)
                self.reset_acceleration_curve_button = self.gui.add_button('Reset Acceleration Curve', visible=self.apply_acceleration_checkbox.value)

            @self.apply_acceleration_checkbox.on_update
            def _(_):
                self.acceleration_fig_plotly.visible = self.apply_acceleration_checkbox.value
                self.acceleration_25_control_point_slider.visible = self.apply_acceleration_checkbox.value
                self.acceleration_50_control_point_slider.visible = self.apply_acceleration_checkbox.value
                self.acceleration_75_control_point_slider.visible = self.apply_acceleration_checkbox.value
                self.reset_acceleration_curve_button.visible = self.apply_acceleration_checkbox.value
                self.editor.logger.info(f'Acceleration from key frame {self._order} to {self._order+1} set to: {self.apply_acceleration_checkbox.value}')

            @self.reset_acceleration_curve_button.on_click
            def _(_):
                self.acceleration_25_control_point_slider.value = 0.1
                self.acceleration_50_control_point_slider.value = 0.50
                self.acceleration_75_control_point_slider.value = 0.9
                self.editor.logger.info(f'Acceleration from key frame {self._order} to {self._order+1} reset')

            def update_acceleration_fig(_):
                if not self.acceleration_fig_plotly.visible:
                    return
                control_points = np.array([
                    0.0,
                    self.acceleration_25_control_point_slider.value,
                    self.acceleration_50_control_point_slider.value,
                    self.acceleration_75_control_point_slider.value,
                    1.0
                ])
                self.acceleration_fig_plotly.figure = get_lut_fig_with_control_points(control_points)
                self.editor.logger.info(f'Acceleration from key frame {self._order} to {self._order+1} set to: {control_points}')
                self.interpolation.acceleration_control_points = control_points

            self.acceleration_25_control_point_slider.on_update(update_acceleration_fig)
            self.acceleration_50_control_point_slider.on_update(update_acceleration_fig)
            self.acceleration_75_control_point_slider.on_update(update_acceleration_fig)

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._folder.order = value*2+101
        self._folder.label = f'Interpolation {value}'

    def __call__(self, net_g, shape):
        return self.interpolation(net_g, shape)

class Interpolation:
    def __init__(self, interval: float, fps: float, keyframe1: KeyFrame, keyframe2: KeyFrame, interval_for_blending: float = 0.1,
                 acceleration_control_points: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]):
        self.interval = interval
        self.fps = fps
        self.keyframe1 = keyframe1
        self.keyframe2 = keyframe2
        self.interval_for_blending = interval_for_blending # for blending hdr and defocus
        self.acceleration_control_points = acceleration_control_points

    @property
    def frame1(self):
        return self.keyframe1.frame

    @property
    def frame2(self):
        return self.keyframe2.frame

    @property
    def need_defocus_blending(self):
        return (self.keyframe1.frame.defocus is None and self.keyframe2.frame.defocus is not None) \
            or (self.keyframe1.frame.defocus is not None and self.keyframe2.frame.defocus is None)

    @property
    def need_hdr_blending(self):
        return (self.keyframe1.frame.hdr != self.keyframe2.frame.hdr)

    @property
    def need_other_interpolation(self):
        if (self.frame1.position != self.frame2.position).any():
            return True
        if (self.frame1.rotation != self.frame2.rotation).any():
            return True
        if self.frame1.fov != self.frame2.fov:
            return True
        if self.frame1.exposure != self.frame2.exposure:
            return True
        if torch.allclose(self.frame1.meta_data[2], self.frame2.meta_data[2]):
            return True
        if self.frame1.color_temp != self.frame2.color_temp:
            return True
        if self.frame1.lut != self.frame2.lut:
            return True
        return False

    def __call__(self, net_g, shape):
        if self.need_defocus_blending or self.need_hdr_blending:
            # need at least 0.1 seconds for blending hdr or defocus
            total_frames_for_blending = int(self.interval_for_blending * self.fps)
            frame1_npy = self.frame1(net_g, shape)
            yield frame1_npy
            frame2 = deepcopy(self.frame1)
            frame2.hdr = self.frame2.hdr
            defocus_init = None if self.frame2.defocus is None else [self.frame2.defocus[0], self.frame2.defocus[1], 9.0]
            frame2.defocus = defocus_init
            frame2_npy = frame2(net_g, shape)
            for i in range(1, total_frames_for_blending):
                t = i / (total_frames_for_blending - 1)
                frame = interpolate(frame1_npy, frame2_npy, t)
                yield frame
            yield frame2_npy
        else:
            yield self.frame1(net_g, shape)

        if self.need_other_interpolation:
            total_frames = int(self.interval * self.fps)

            pose1 = self.frame1.pose
            pose2 = self.frame2.pose
            fov1 = self.frame1.fov
            fov2 = self.frame2.fov
            exposure1 = self.frame1.exposure
            exposure2 = self.frame2.exposure
            color_temp1 = self.frame1.color_temp
            color_temp2 = self.frame2.color_temp
            lut1 = self.frame1.lut
            lut2 = self.frame2.lut
            meta_data1 = self.frame1.meta_data
            meta_data2 = self.frame2.meta_data
            defocus1 = defocus_init if self.need_defocus_blending else self.frame1.defocus
            defocus2 = self.frame2.defocus

            hdr = self.frame2.hdr
            full_hdr = self.frame2.full_hdr
            poses = interpolate_poses_with_acceleration(pose1, pose2, total_frames, self.acceleration_control_points)

            for t in range(1, total_frames):
                pose = poses[t-1]
                fov = interpolate(fov1, fov2, t / total_frames)
                if exposure1 is not None and exposure2 is not None:
                    exposure = interpolate(exposure1, exposure2, t / total_frames)
                    color_temp = interpolate(color_temp1, color_temp2, t / total_frames)
                    wb = interpolate(meta_data1[2], meta_data2[2], t / total_frames)
                    meta_data = deepcopy(meta_data1)
                    meta_data[2] = wb
                else:
                    exposure = None
                    color_temp = None
                    meta_data = None
                lut = interpolate(lut1, lut2, t / total_frames)
                if defocus1 is not None and defocus2 is not None:
                    plane = interpolate(defocus1[0], defocus2[0], t / total_frames)
                    near_far = interpolate(np.array(defocus1[1]), np.array(defocus2[1]), t / total_frames)
                    aperture = interpolate(defocus1[2], defocus2[2], t / total_frames)
                    defocus = (plane, near_far, aperture)
                else:
                    defocus = None

                position, rotation = pose[:3, 3], pose[:3, :3]
                frame = Frame(position, rotation, fov, exposure, meta_data, hdr, full_hdr, color_temp, lut, defocus)
                yield frame(net_g, shape)

            yield self.frame2(net_g, shape)

def numpy_gamma_expansion(srgb):
    eps = np.finfo(np.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = np.maximum(eps, ((200 * srgb + 11) / (211))**(12 / 5))
    return np.where(srgb <= 0.04045, linear0, linear1)

class FrameInterpolater:
    def __init__(self, interpolations, return_linear=False):
        self.interpolations = interpolations
        self.return_linear = return_linear

    def __call__(self, net_g, shape):
        for ii, interpolation in enumerate(self.interpolations):
            for jj, frame in enumerate(interpolation(net_g, shape)):
                if ii > 0 and jj == 0:
                    # skip the first frame of each interpolation to avoid duplicate frames
                    continue
                yield frame if not self.return_linear else numpy_gamma_expansion(frame)
