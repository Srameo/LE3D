import math
import time
import numpy as np
import torch
from os import path as osp
from basicgs.data.camera_utils import Camera
from basicgs.viewers.base_viewer import BaseViewer
from basicgs.viewers.le3d_viewer import Le3dViewer
from basicgs.viewers.utils import get_w2c, get_wxyz
from basicgs.viewers.utils.export_util import Exporter
from basicgs.viewers.utils.frame_util import KeyFrame
from basicgs.viewers.utils.interpolate_util import FrameInterpolater, KeyInterpolation

def pad_to_match_aspect(image, aspect):
    h, w, c = image.shape
    new_w = int(h * aspect)
    if new_w > w:
        pad_w = (new_w - w) // 2
        return np.pad(image, ((0, 0), (pad_w, pad_w), (0, 0)), mode='constant')
    else:
        new_h = int(w / aspect)
        pad_h = (new_h - h) // 2
        return np.pad(image, ((pad_h, pad_h), (0, 0), (0, 0)), mode='constant')

def mask_to_match_aspect(image, aspect):
    h, w, c = image.shape
    mask = np.ones_like(image) * 0.5
    if w * aspect > h:
        # pad w
        pad_w = int((w - h / aspect) / 2)
        mask[:, pad_w:-pad_w, :] = 0.0
    else:
        # pad h
        pad_h = int((h - w * aspect) / 2)
        mask[pad_h:-pad_h, :, :] = 0.0
    return (1 - mask) * image + mask * 1

class Editor:
    def __init__(self, viewer: BaseViewer):
        self.viewer = viewer
        self.key_frames = []
        self.interpolations = []
        self._preview_camera = None
        self._preview_interpolation = None

        self.exporter = None
        self.init_components()

    def reset_order(self):
        for idx, (kf, inter) in enumerate(zip(self.key_frames, self.interpolations + [None])):
            kf.order = idx
            if inter is not None:
                inter.order = idx

    def delete_key_frame(self, order):
        self._preview_camera = None
        self._preview_interpolation = None
        self.continue_button.visible = False

        if order == len(self.key_frames)-1:
            interp = self.interpolations.pop(order-1)
            interp._folder.visible = False
            del interp
        elif order == 0:
            if len(self.interpolations) > 0:
                interp = self.interpolations.pop(0)
                interp._folder.visible = False
            del interp
        else:
            interp = self.interpolations.pop(order)
            interp._folder.visible = False
            del interp
            interp = self.interpolations.pop(order-1)
            interp._folder.visible = False
            del interp
            interp = KeyInterpolation(self, order-1, self.gui, fps=float(self.fps.value), keyframe1=self.key_frames[order-1], keyframe2=self.key_frames[order+1])
            self.interpolations.insert(order-1, interp)

        keyframe = self.key_frames.pop(order)
        keyframe._folder.visible = False
        del keyframe

        self.reset_order()

    @property
    def preview_camera(self):
        return self._preview_camera

    @preview_camera.setter
    def preview_camera(self, value):
        self._preview_camera = value
        if value is not None:
            self.continue_button.label = f'Previewing Key Frame {value}, click to continue'
            self.continue_button.visible = True
        else:
            self.continue_button.visible = False

    @property
    def preview_interpolation(self):
        return self._preview_interpolation

    @preview_interpolation.setter
    def preview_interpolation(self, value):
        self._preview_interpolation = value
        if value is not None:
            self.continue_button.label = f'Previewing Interpolation {value}, click to continue'
            self.continue_button.visible = True
        else:
            self.continue_button.visible = False

    @property
    def gui(self):
        return self.viewer.viser_server.gui

    @property
    def logger(self):
        return self.viewer.logger

    def init_components(self):
        with self.viewer.tabs.add_tab('Editor') as tab:
            self.editor_tab = tab
            self.fps = self.gui.add_text('FPS:', '30')
            self.continue_button = self.gui.add_button('Continue to View', hint='Continue to the view', order=-1, visible=False)
            self.add_key_frame_button = self.gui.add_button('Add Key Frame', hint='Add a key frame', order=-1)

        @self.continue_button.on_click
        def _(_):
            self._preview_camera = None
            self._preview_interpolation = None
            self.continue_button.visible = False

        @self.add_key_frame_button.on_click
        def _(_):
            w2c = get_w2c(self.current_camera)
            if isinstance(self.viewer, Le3dViewer):
                meta_data = self.viewer.meta_data
                meta_data[2] = torch.tensor(self.viewer.wb_value_array.value, device='cuda')
                isp_params = dict(
                    exposure=self.viewer.exposure_value_slider.value,
                    meta_data=meta_data,
                    color_temp=self.viewer.color_temp_slider.value,
                    hdr=self.viewer.hdr_checkbox.value,
                    full_hdr=self.viewer.full_hdr_checkbox.value,
                )
            else:
                isp_params = dict()
            defocus = (
                self.viewer.defocus_plane_slider.value,
                self.viewer.defocus_near_far_slider.value,
                self.viewer.defocus_aperture_slider.value,
            ) if self.viewer.defocus_checkbox.value else None
            with self.editor_tab:
                keyframe = KeyFrame(
                    editor=self,
                    order=len(self.key_frames),
                    gui=self.gui,
                    position=w2c[:3, 3],
                    rotation=w2c[:3,:3],
                    fov=self.viewer.fov_slider.value,
                    lut=self.viewer.get_lut_control_points(),
                    defocus=defocus,
                    **isp_params,
                )
                self.key_frames.append(keyframe)
                if len(self.key_frames) > 1:
                    interpolation = KeyInterpolation(
                        editor=self,
                        order=len(self.interpolations),
                        gui=self.gui,
                        fps=float(self.fps.value),
                        keyframe1=self.key_frames[-2],
                        keyframe2=self.key_frames[-1],
                    )
                    self.interpolations.append(interpolation)
            self.logger.info(f'Key frame {len(self.key_frames)-1} added: {self.key_frames[-1].json}')

        with self.viewer.tabs.add_tab('Export') as tab:
            self.export_tab = tab
            self.export_path_text = self.gui.add_text('Export Path:', hint='The path to export the video', initial_value=f'output/demo_video/{osp.basename(self.viewer.exp_opt["name"])}.mp4')
            self.export_resolution_vector = self.gui.add_vector2('Resolution', hint='The resolution of the exported video', initial_value=(1920, 1080))
            self.export_button = self.gui.add_button('Export', hint='Export the current key frames and interpolations')

            @self.export_button.on_click
            def _(_):
                self.logger.info(f'Exporting to video to {self.export_path_text.value}...')
                self.exporter = Exporter(file_path=self.export_path_text.value, fps=float(self.fps.value),
                                         resolution=list(map(int, self.export_resolution_vector.value[::-1])))

    @property
    def export_aspect(self):
        return float(self.export_resolution_vector.value[1]) / float(self.export_resolution_vector.value[0])

    @torch.no_grad()
    def update(self):
        clients = list(self.viewer.viser_server.get_clients().values())
        if not self.viewer.need_update or len(clients) == 0:
            return

        client = clients[-1]
        camera = client.camera
        self.current_camera = camera
        W = self.viewer.resolution_slider.value
        H = int(self.viewer.resolution_slider.value/camera.aspect)

        if self.exporter is not None:
            frame_interpolater = FrameInterpolater(self.interpolations, return_linear=False)
            for frame in frame_interpolater(self.viewer.net_g,
                                            list(map(int, self.export_resolution_vector.value[::-1]))):
                self.exporter.add_frame(frame)
                frame = pad_to_match_aspect(frame, camera.aspect)
                client.set_background_image(frame, format="jpeg")
            self.exporter.close()
            self.exporter = None

        if self.preview_camera is None and self.preview_interpolation is None:
            if self.viewer.tuning_camera_checkbox.value:
                camera.wxyz = get_wxyz(
                    np.array(self.viewer.camera_position_vec3.value),
                    np.array(self.viewer.camera_look_at_vec3.value),
                    np.array(self.viewer.camera_up_vec3.value)
                )
            else:
                self.viewer.camera_position_vec3.value = camera.position
                self.viewer.camera_up_vec3.value = camera.up_direction
                self.viewer.camera_look_at_vec3.value = camera.look_at

            w2c = get_w2c(camera)

            camera.fov = self.viewer.fov_slider.value / 180.0 * math.pi
            fov_x = camera.fov * camera.aspect / 2
            fov_y = camera.fov / 2

            camera_pack = Camera(colmap_id=0, R=w2c[:3,:3], T=w2c[:3, 3], fov_x=fov_x, fov_y=fov_y,
                                image=torch.zeros((3, H, W)), gt_alpha_mask=None,
                                image_name='viser_viewer_fake_img.jpg', uid=0,
                                znear=0.2, zfar=1000,
                                bayer_mask=False)
            camera_pack = camera_pack.to('cuda')
            rendered_image, render_time, _ = self.viewer.get_current_image(camera_pack)
            rendered_image = mask_to_match_aspect(rendered_image, self.export_aspect)
            client.set_background_image(rendered_image, format="jpeg")
        elif self.preview_camera is not None:
            rendered_image = self.key_frames[self.preview_camera](self.viewer.net_g, (H, W))
            rendered_image = mask_to_match_aspect(rendered_image, self.export_aspect)
            client.set_background_image(rendered_image, format="jpeg")
        else:
            start_time = time.time()
            fps_time = 1/float(self.fps.value)
            for rendered_image in self.interpolations[self.preview_interpolation](self.viewer.net_g, (H, W)):
                end_time = time.time()
                if end_time - start_time < fps_time:
                    time.sleep(fps_time - (end_time - start_time))
                rendered_image = mask_to_match_aspect(rendered_image, self.export_aspect)
                client.set_background_image(rendered_image, format="jpeg")
                start_time = time.time()
            self.preview_interpolation = None

