from collections import deque
import math
import time
import numpy as np
import torch
import viser
from viser import transforms as tf
from basicgs.data.camera_utils import Camera
from basicgs.utils import get_time_str, get_root_logger
from basicgs.utils.raw_util import apply_lut, apply_defocus
from basicgs.utils.registry import VIEWER_REGISTRY
from basicgs.viewers.utils import get_lut, get_w2c, depth_naninf_to_red, get_wxyz, get_lut_fig_with_control_points
from basicgs.gaussians.util.render import render, render_hist, render_full_package
import scipy.interpolate as interp

NEAR_FAR_DEFAULT = (0.2, 400)

@VIEWER_REGISTRY.register()
class BaseViewer:
    @property
    def user_interactive_mode(_super):
        class UserInteractiveMode:
            def __enter__(self):
                self.need_update_bak = _super.need_update
                _super.need_update = False
            def __exit__(self, exc_type, exc_value, traceback):
                _super.need_update = self.need_update_bak
        return UserInteractiveMode()

    @property
    def need_render_image(self):
        return not self.render_depth_checkbox.value and \
               not self.render_final_opacity_checkbox.value and \
               not self.render_hist_checkbox.value

    def __init__(self, exp_opt, net_g, log_dir, update_freq=None, total_iters=None, port=8097) -> None:
        self.exp_opt = exp_opt
        self.port = port
        self.viser_server = viser.ViserServer(port=port)
        self.need_update = False
        self.logger = get_root_logger('basicgs.viewer', log_file=f'{log_dir}/viewer_{get_time_str()}.log')
        self.logger.info(f'Viewer initialized, port: {port}')
        if total_iters is not None:
            self.logger.info(f'Initializing viewer in training mode with {total_iters} iterations')
        self.tabs = self.viser_server.gui.add_tab_group()
        self.current_iter = 0
        self.update_freq = update_freq
        self.net_g = net_g
        self.total_iters = total_iters
        self.render_time_list = deque(maxlen=30)
        self.log_ema_dict = {}
        self.bg_color_override = None
        self.current_click_pos = None

        self.init_components()
        self.need_update = True

    def set_current_iter(self, current_iter, log_dict):
        self.current_iter = current_iter
        if self.total_iters is not None:
            self.progress_bar_text.value = f'{self.current_iter}/{self.total_iters}, update freq: {self.update_freq}'
        if log_dict is not None:
            log_str = ''
            for k, v in log_dict.items():
                if k not in self.log_ema_dict:
                    self.log_ema_dict[k] = v
                else:
                    self.log_ema_dict[k] = self.log_ema_dict[k] * 0.6 + v * 0.4
                log_str += '<div style={{ fontSize: "12px" }}>' + f'{k}: {log_dict[k]:.4e} ({self.log_ema_dict[k]:.4e})</div>\n'
            self.log_markdown.content = log_str

    def update_gaussians(self, net_g):
        if self.update_freq is not None and self.update_freq % self.current_iter == 0:
            with self.user_interactive_mode:
                self.net_g = net_g

    def init_components(self):
        default_look_at = np.array((1, 0, 0), dtype=np.float32)
        default_position = np.array((1, 15, -100), dtype=np.float32)
        default_up = np.array((0, -1, 0), dtype=np.float32)
        with self.tabs.add_tab('Basic') as tab:
            self.basic_tab = tab
            if self.total_iters is not None:
                with self.viser_server.gui.add_folder('Training'):
                    self.focus_on_training_button = self.viser_server.gui.add_button('Focus on Training (Stop Rendering)')
                    self.progress_bar_text = self.viser_server.gui.add_text('Progress', initial_value=f'0/{self.total_iters}', disabled=True)
                    with self.viser_server.gui.add_folder('Log'):
                        self.log_markdown = self.viser_server.gui.add_markdown('log here, with ema 0.6')
            with self.viser_server.gui.add_folder('Rendering'):
                self.render_time_text = self.viser_server.gui.add_text('Render Time', initial_value='0.00 ms', disabled=True,
                                                                   hint='Average render time of the last 30 frames, this contains the time of isp or other post-processing')
                self.resolution_slider = self.viser_server.gui.add_slider('Resolution', min=512, max=4096, step=2, initial_value=1024)
                self.fov_slider = self.viser_server.gui.add_slider('FOV', min=10, max=100, step=1, initial_value=65.5)
                self.background_color_vector3 = self.viser_server.gui.add_vector3('Background Color', initial_value=(0.0, 0.0, 0.0), step=0.01, max=(1.0, 1.0, 1.0), min=(0.0, 0.0, 0.0))
                with self.viser_server.gui.add_folder('Pose'):
                    self.camera_look_at_vec3 = self.viser_server.gui.add_vector3("Look At", default_look_at, step=0.001, disabled=True)
                    self.camera_position_vec3 = self.viser_server.gui.add_vector3("Position", default_position, step=0.001, disabled=True)
                    self.camera_up_vec3 = self.viser_server.gui.add_vector3("Up", default_up, step=0.001, disabled=True)
                    self.tuning_camera_checkbox = self.viser_server.gui.add_checkbox('Tuning Camera', False)
                    self.reset_camera_button = self.viser_server.gui.add_button('Reset Camera', visible=self.tuning_camera_checkbox.value)
                with self.viser_server.gui.add_folder('Depth'):
                    self.render_depth_checkbox = self.viser_server.gui.add_checkbox('Depth', False)
                    self.depth_near_far_slider = self.viser_server.gui.add_multi_slider('Near/Far', min=NEAR_FAR_DEFAULT[0], max=NEAR_FAR_DEFAULT[1], step=0.1, initial_value=NEAR_FAR_DEFAULT, visible=self.render_depth_checkbox.value)
                with self.viser_server.gui.add_folder('Final Opacity'):
                    self.render_final_opacity_checkbox = self.viser_server.gui.add_checkbox('Final Opacity', False)
                with self.viser_server.gui.add_folder('Histogram'):
                    self.render_hist_checkbox = self.viser_server.gui.add_checkbox('Histogram', False)
                    self.hist_normalize_checkbox = self.viser_server.gui.add_checkbox('Normalize', False, visible=self.render_hist_checkbox.value)
                    self.hist_plane_slider = self.viser_server.gui.add_slider('Histogram Bins', min=1, max=32, step=1, initial_value=1, visible=self.render_hist_checkbox.value)
                    self.hist_near_far_slider = self.viser_server.gui.add_multi_slider('Near/Far', min=NEAR_FAR_DEFAULT[0], max=NEAR_FAR_DEFAULT[1], step=0.1, initial_value=NEAR_FAR_DEFAULT, visible=self.render_hist_checkbox.value)

        if self.total_iters is not None:
            @self.focus_on_training_button.on_click
            def _(_):
                if self.need_update:
                    self.need_update = False
                    self.focus_on_training_button.label = 'Continue Rendering'
                else:
                    self.need_update = True
                    self.focus_on_training_button.label = 'Focus on Training (Stop Rendering)'

        @self.render_depth_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.depth_near_far_slider.visible = self.render_depth_checkbox.value
                if self.render_depth_checkbox.value:
                    self.render_final_opacity_checkbox.value = False
                    self.render_hist_checkbox.value = False

        @self.render_final_opacity_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                if self.render_final_opacity_checkbox.value:
                    self.render_depth_checkbox.value = False
                    self.render_hist_checkbox.value = False

        @self.render_hist_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.hist_plane_slider.visible = self.render_hist_checkbox.value
                self.hist_near_far_slider.visible = self.render_hist_checkbox.value
                self.hist_normalize_checkbox.visible = self.render_hist_checkbox.value
                if self.render_hist_checkbox.value:
                    self.render_final_opacity_checkbox.value = False
                    self.render_depth_checkbox.value = False

        @self.background_color_vector3.on_update
        def _(_):
            with self.user_interactive_mode:
                self.bg_color_override = torch.tensor(self.background_color_vector3.value,
                                                      device=self.net_g.bg_color.device).float()

        @self.tuning_camera_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.tuning_camera_checkbox.value = False
                self.camera_look_at_vec3.disabled = not self.tuning_camera_checkbox.value
                self.camera_position_vec3.disabled = not self.tuning_camera_checkbox.value
                self.camera_up_vec3.disabled = not self.tuning_camera_checkbox.value
                self.reset_camera_button.visible = self.tuning_camera_checkbox.value
                self.tuning_camera_checkbox.value = True

        @self.reset_camera_button.on_click
        def _(_):
            with self.user_interactive_mode:
                self.camera_look_at_vec3.value = default_look_at
                self.camera_position_vec3.value = default_position
                self.camera_up_vec3.value = default_up

        with self.tabs.add_tab('Post-Processing') as tab:
            self.post_processing_tab = tab
            with self.viser_server.gui.add_folder('Enhancement'):
                self.apply_lut_checkbox = self.viser_server.gui.add_checkbox('Curve Enhance', False)
                self.lut_fig_plotly = self.viser_server.gui.add_plotly(get_lut_fig_with_control_points(np.array([0.0, 0.25, 0.5, 0.75, 1.0])), aspect=1.0, visible=self.apply_lut_checkbox.value)
                self.lut_25_control_point_slider = self.viser_server.gui.add_slider('Control Point (0.25)', min=0, max=1, step=0.01, initial_value=0.25, visible=self.apply_lut_checkbox.value)
                self.lut_50_control_point_slider = self.viser_server.gui.add_slider('Control Point (0.50)', min=0, max=1, step=0.01, initial_value=0.50, visible=self.apply_lut_checkbox.value)
                self.lut_75_control_point_slider = self.viser_server.gui.add_slider('Control Point (0.75)', min=0, max=1, step=0.01, initial_value=0.75, visible=self.apply_lut_checkbox.value)
                self.reset_lut_curve_button = self.viser_server.gui.add_button('Reset LUT Curve', visible=self.apply_lut_checkbox.value)
            with self.viser_server.gui.add_folder('Defocus'):
                self.defocus_checkbox = self.viser_server.gui.add_checkbox('Defocus', False)
                self.defocus_clickable_checkbox = self.viser_server.gui.add_checkbox('Clickable', False, visible=self.defocus_checkbox.value)
                self.defocus_near_far_slider = self.viser_server.gui.add_multi_slider('Near/Far', min=NEAR_FAR_DEFAULT[0], max=NEAR_FAR_DEFAULT[1], step=0.1, initial_value=NEAR_FAR_DEFAULT, visible=self.defocus_checkbox.value)
                self.defocus_plane_slider = self.viser_server.gui.add_slider('Plane', min=0, max=1, step=0.01, initial_value=0.0, visible=self.defocus_checkbox.value)
                self.defocus_aperture_slider = self.viser_server.gui.add_slider('Aperture', min=0.5, max=9.0, step=0.01, initial_value=9.0, visible=self.defocus_checkbox.value)

        @self.apply_lut_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.lut_fig_plotly.visible = self.apply_lut_checkbox.value
                self.lut_25_control_point_slider.visible = self.apply_lut_checkbox.value
                self.lut_50_control_point_slider.visible = self.apply_lut_checkbox.value
                self.lut_75_control_point_slider.visible = self.apply_lut_checkbox.value
                self.reset_lut_curve_button.visible = self.apply_lut_checkbox.value

        @self.reset_lut_curve_button.on_click
        def _(_):
            with self.user_interactive_mode:
                self.lut_25_control_point_slider.value = 0.25
                self.lut_50_control_point_slider.value = 0.50
                self.lut_75_control_point_slider.value = 0.75

        def update_lut_fig(_):
            if not self.lut_fig_plotly.visible:
                return
            control_points = self.get_lut_control_points()
            if control_points is None:
                return
            self.lut_fig_plotly.figure = get_lut_fig_with_control_points(control_points)

        self.lut_25_control_point_slider.on_update(update_lut_fig)
        self.lut_50_control_point_slider.on_update(update_lut_fig)
        self.lut_75_control_point_slider.on_update(update_lut_fig)

        @self.defocus_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.defocus_clickable_checkbox.visible = self.defocus_checkbox.value
                self.defocus_near_far_slider.visible = self.defocus_checkbox.value
                self.defocus_plane_slider.visible = self.defocus_checkbox.value
                self.defocus_aperture_slider.visible = self.defocus_checkbox.value

        @self.defocus_clickable_checkbox.on_update
        def _(_):
            self.defocus_plane_slider.disabled = self.defocus_clickable_checkbox.value
            if self.defocus_clickable_checkbox.value:
                @self.viser_server.scene.on_pointer_event('click')
                def _(event: viser.ScenePointerEvent):
                    camera = event.client.camera
                    ray_d = torch.Tensor(tf.SO3(camera.wxyz).inverse().as_matrix() @ event.ray_direction)
                    # ray_d, the ray in the camera, x->right, y->down, z->far
                    xy_norm = ray_d[:2]/ray_d[2]

                    WH = torch.Tensor([self.resolution_slider.value, self.resolution_slider.value/camera.aspect]).int()
                    tan_fov_2 = torch.Tensor([np.tan(camera.fov/2)*camera.aspect, np.tan(camera.fov/2)])
                    fxy = WH/2 / tan_fov_2

                    uv = (fxy * xy_norm  + WH/2).int()
                    self.current_click_pos = uv
                    self.logger.info(f'clicking: {self.current_click_pos.numpy()} in {WH.numpy()}')
            else:
                self.viser_server.scene.remove_pointer_callback()
                self.current_click_pos = None

    def get_lut_control_points(self):
        lut_25 = self.lut_25_control_point_slider.value
        lut_50 = self.lut_50_control_point_slider.value
        lut_75 = self.lut_75_control_point_slider.value
        return np.array([0.0, lut_25, lut_50, lut_75, 1.0])

    @torch.no_grad()
    def get_current_image(self, camera, require_other_process=False):
        snapshot = {
            'render_depth': self.render_depth_checkbox.value,
            'render_final_opacity': self.render_final_opacity_checkbox.value,
            'render_hist': self.render_hist_checkbox.value,
            'render_image': self.need_render_image
        }
        start_time = time.time()
        if snapshot['render_hist']:
            near_far = self.hist_near_far_slider.value
            render_pkg = render_hist(self.net_g, camera, num_bins=32, near_far=near_far)
            rendered_image = render_pkg['render']
            if self.hist_normalize_checkbox.value:
                rendered_image = rendered_image / (render_pkg['render'].sum(dim=0, keepdim=True) + 1.0e-3)
            rendered_image = rendered_image[self.hist_plane_slider.value-1][None].repeat(3, 1, 1) # for visualization, turn in to rgb
        elif snapshot['render_final_opacity']:
            render_pkg = render_full_package(self.net_g, camera)
            rendered_image = render_pkg['final_opacity']
            rendered_image = rendered_image.repeat(3, 1, 1)
        elif snapshot['render_depth']:
            render_pkg = render_full_package(self.net_g, camera)
            rendered_image = render_pkg['depth'] / (render_pkg['final_opacity'] + 1.0e-3)
            rendered_image = (rendered_image - self.depth_near_far_slider.value[0]) / (self.depth_near_far_slider.value[1] - self.depth_near_far_slider.value[0])
            rendered_image = depth_naninf_to_red(rendered_image)
        else:
            render_pkg = render(self.net_g, camera, bg_color_override=self.bg_color_override)
            rendered_image = render_pkg['render']
        if not require_other_process:
            if self.defocus_checkbox.value:
                target_bin = round(self.defocus_plane_slider.value * 31)
                hists = render_hist(self.net_g, camera, num_bins=32, near_far=self.defocus_near_far_slider.value)
                rendered_pkg = apply_defocus(rendered_image, hists, target_bin, 32,
                                             delta_r=1/self.defocus_aperture_slider.value,
                                             position=self.current_click_pos)
                rendered_image = rendered_pkg[0]
                if self.current_click_pos is not None:
                    self.defocus_plane_slider.value = rendered_pkg[1]
            # apply lut in rgb space
            if self.apply_lut_checkbox.value:
                lut = get_lut(self.get_lut_control_points(), 16)
                rendered_image = apply_lut(rendered_image, lut, 16)
            torch.cuda.synchronize()
            end_time = time.time()
            rendered_image = rendered_image.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        else:
            torch.cuda.synchronize()
            end_time = time.time()
        return rendered_image, end_time - start_time, snapshot

    @property
    def avg_render_time(self):
        return sum(self.render_time_list) / len(self.render_time_list)

    @torch.no_grad()
    def update(self):
        clients = list(self.viser_server.get_clients().values())
        if not self.need_update or len(clients) == 0:
            return

        client = clients[-1]
        camera = client.camera

        if self.tuning_camera_checkbox.value:
            camera.wxyz = get_wxyz(
                np.array(self.camera_position_vec3.value),
                np.array(self.camera_look_at_vec3.value),
                np.array(self.camera_up_vec3.value)
            )
        else:
            self.camera_position_vec3.value = camera.position
            self.camera_up_vec3.value = camera.up_direction
            self.camera_look_at_vec3.value = camera.look_at

        camera.fov = self.fov_slider.value / 180.0 * math.pi
        w2c = get_w2c(camera)

        W = self.resolution_slider.value
        H = int(self.resolution_slider.value/camera.aspect)
        fov_x = camera.fov * camera.aspect / 2
        fov_y = camera.fov / 2

        camera_pack = Camera(colmap_id=0, R=w2c[:3,:3], T=w2c[:3, 3], fov_x=fov_x, fov_y=fov_y,
                             image=torch.zeros((3, H, W)), gt_alpha_mask=None,
                             image_name='viser_viewer_fake_img.jpg', uid=0,
                             znear=0.2, zfar=1000,
                             bayer_mask=False)
        camera_pack = camera_pack.to('cuda')
        rendered_image, render_time, _ = self.get_current_image(camera_pack)
        self.render_time_list.append(render_time)
        self.render_time_text.value = f'{self.avg_render_time*1000:8.2f} ms; {1/self.avg_render_time:8.2f} FPS'
        client.set_background_image(rendered_image, format="jpeg")

    def __del__(self):
        self.viser_server.stop()
