from copy import deepcopy
import math
import numpy as np
import torch
from viser import GuiApi
from basicgs.data.camera_utils import Camera
from basicgs.gaussians.util.render import render, render_hist
from basicgs.utils.raw_util import adjust_color_temperature, apply_defocus, apply_lut, finish_isp, hdr_plus_photo_finish
from basicgs.viewers.utils import get_lut, get_lut_fig_with_control_points, get_w2c
from basicgs.viewers.base_viewer import NEAR_FAR_DEFAULT

DEVICE = torch.device('cuda')

class KeyFrame:
    def __init__(self, editor, order, gui: GuiApi, position: np.ndarray, rotation: np.ndarray, fov: float, exposure: float = None, meta_data: list[torch.Tensor] = None, hdr: bool = None, full_hdr: bool = None, color_temp: float = None, lut: np.ndarray = None, defocus: list = None):
        self.editor = editor
        self.frame: Frame = Frame(position, rotation, fov, exposure, meta_data, hdr, full_hdr, color_temp, lut, defocus)
        self.gui: GuiApi = gui
        self._order = order
        self._folder = self.gui.add_folder(f'Key Frame {order}', order=order*2+100, expand_by_default=False)

        with self._folder:
            self.view_button = self.gui.add_button('View', hint='View this key frame')
            self.delete_button = self.gui.add_button('Delete', hint='Delete this key frame')

            @self.view_button.on_click
            def _(_):
                self.editor.preview_interpolation = None
                self.editor.preview_camera = self.order
                self.editor.logger.info(f'Preview key frame {self._order}')

            @self.delete_button.on_click
            def _(_):
                self.editor.delete_key_frame(self.order)

            with self.gui.add_folder('Camera'):
                self.cam_pos_reset_button = self.gui.add_button('Reset Cam Pos', hint='Reset Position to current position')
                @self.cam_pos_reset_button.on_click
                def _(_):
                    w2c = get_w2c(self.editor.current_camera)
                    self.frame.position = w2c[:3, 3]
                    self.frame.rotation = w2c[:3, :3]
                    self.editor.logger.info(f'Camera pos of key frame {self._order} reset to position: {self.frame.position}')
                    self.editor.logger.info(f'Camera pos of key frame {self._order} reset to rotation: {self.frame.rotation}')
                    self.editor.preview_camera = self.order

                self.fov_slider = self.gui.add_slider('FOV', min=10, max=100, step=1, initial_value=self.frame.fov)
                @self.fov_slider.on_update
                def _(_):
                    self.frame.fov = self.fov_slider.value
                    self.editor.logger.info(f'Camera fov of key frame {self._order} set to: {self.frame.fov}')

            # only for isp
            if exposure is not None:
                self._isp_folder = self.gui.add_folder('ISP') if not hasattr(self, '_isp_folder') else self._isp_folder
                with self._isp_folder:
                    self.exposure_slider = self.gui.add_slider('Exposure Value', -8, 8, 0.01, exposure)

                @self.exposure_slider.on_update
                def _(_):
                    self.frame.exposure = self.exposure_slider.value
                    self.editor.logger.info(f'ISP exposure of key frame {self._order} set to: {self.frame.exposure}')

            if meta_data is not None:
                self._isp_folder = self.gui.add_folder('ISP') if not hasattr(self, '_isp_folder') else self._isp_folder
                with self._isp_folder:
                    wb = meta_data[2].cpu().numpy()
                    self.wb_value_array = self.gui.add_vector3('White Balance', wb)

                @self.wb_value_array.on_update
                def _(_):
                    self.frame.meta_data[2] = torch.tensor(self.wb_value_array.value, device=DEVICE)
                    self.editor.logger.info(f'ISP white balance of key frame {self._order} set to: {self.wb_value_array.value}')

            if color_temp is not None:
                self._isp_folder = self.gui.add_folder('ISP') if not hasattr(self, '_isp_folder') else self._isp_folder
                with self._isp_folder:
                    self.color_temp_slider = self.gui.add_slider('Color Temp', 2000, 40000, 100, color_temp)

                @self.color_temp_slider.on_update
                def _(_):
                    self.frame.color_temp = self.color_temp_slider.value
                    self.editor.logger.info(f'ISP color temp of key frame {self._order} set to: {self.frame.color_temp}')

            if hdr is not None:
                self._isp_folder = self.gui.add_folder('ISP') if not hasattr(self, '_isp_folder') else self._isp_folder
                with self._isp_folder:
                    self.hdr_checkbox = self.gui.add_checkbox('HDR', hdr)
                    self.full_hdr_checkbox = self.gui.add_checkbox('Full HDR', False, visible=self.hdr_checkbox.value)

                @self.hdr_checkbox.on_update
                def _(_):
                    self.full_hdr_checkbox.visible = self.hdr_checkbox.value
                    self.frame.hdr = self.hdr_checkbox.value
                    self.editor.logger.info(f'ISP hdr of key frame {self._order} set to: {self.frame.hdr}')

                @self.full_hdr_checkbox.on_update
                def _(_):
                    self.frame.full_hdr = self.full_hdr_checkbox.value
                    self.editor.logger.info(f'ISP full hdr of key frame {self._order} set to: {self.frame.full_hdr}')

            with self.gui.add_folder('Enhancement'):
                self.apply_lut_checkbox = self.gui.add_checkbox('Curve Enhance', lut is not None)
                self.lut_fig_plotly = self.gui.add_plotly(get_lut_fig_with_control_points(lut), aspect=1.0, visible=self.apply_lut_checkbox.value)
                init_lut_25, init_lut_50, init_lut_75 = lut[1], lut[2], lut[3]
                self.lut_25_control_point_slider = self.gui.add_slider('Control Point (0.25)', min=0, max=1, step=0.01, initial_value=init_lut_25, visible=self.apply_lut_checkbox.value)
                self.lut_50_control_point_slider = self.gui.add_slider('Control Point (0.50)', min=0, max=1, step=0.01, initial_value=init_lut_50, visible=self.apply_lut_checkbox.value)
                self.lut_75_control_point_slider = self.gui.add_slider('Control Point (0.75)', min=0, max=1, step=0.01, initial_value=init_lut_75, visible=self.apply_lut_checkbox.value)
                self.reset_lut_curve_button = self.gui.add_button('Reset LUT Curve', visible=self.apply_lut_checkbox.value)

                @self.apply_lut_checkbox.on_update
                def _(_):
                    self.lut_fig_plotly.visible = self.apply_lut_checkbox.value
                    self.lut_25_control_point_slider.visible = self.apply_lut_checkbox.value
                    self.lut_50_control_point_slider.visible = self.apply_lut_checkbox.value
                    self.lut_75_control_point_slider.visible = self.apply_lut_checkbox.value
                    self.reset_lut_curve_button.visible = self.apply_lut_checkbox.value
                    self.editor.logger.info(f'Curve enhance of key frame {self._order} set to: {self.apply_lut_checkbox.value}')

                @self.reset_lut_curve_button.on_click
                def _(_):
                    self.lut_25_control_point_slider.value = 0.25
                    self.lut_50_control_point_slider.value = 0.50
                    self.lut_75_control_point_slider.value = 0.75
                    self.editor.logger.info(f'Curve enhance of key frame {self._order} reset')

                def update_lut_fig(_):
                    if not self.lut_fig_plotly.visible:
                        return
                    control_points = np.array([
                        0.0,
                        self.lut_25_control_point_slider.value,
                        self.lut_50_control_point_slider.value,
                        self.lut_75_control_point_slider.value,
                        1.0
                    ])
                    self.lut_fig_plotly.figure = get_lut_fig_with_control_points(control_points)
                    self.frame.lut = control_points
                    self.editor.logger.info(f'Curve control points of key frame {self._order} set to: {control_points}')

                self.lut_25_control_point_slider.on_update(update_lut_fig)
                self.lut_50_control_point_slider.on_update(update_lut_fig)
                self.lut_75_control_point_slider.on_update(update_lut_fig)

            with self.gui.add_folder('Defocus'):
                self.defocus_checkbox = self.gui.add_checkbox('Defocus', defocus is not None)
                defocus_plane, defocus_near_far, defocus_aperture = defocus if defocus is not None else (0.0, NEAR_FAR_DEFAULT, 9.0)
                self.defocus_plane_slider = self.gui.add_slider('Defocus Plane', min=0, max=1, step=0.01, initial_value=defocus_plane, visible=self.defocus_checkbox.value)
                self.defocus_near_far_slider = self.gui.add_multi_slider('Near Far', min=NEAR_FAR_DEFAULT[0], max=NEAR_FAR_DEFAULT[1], step=0.01, initial_value=defocus_near_far, visible=self.defocus_checkbox.value)
                self.defocus_aperture_slider = self.gui.add_slider('Aperture', min=0.5, max=9.0, step=0.01, initial_value=defocus_aperture, visible=self.defocus_checkbox.value)

                @self.defocus_checkbox.on_update
                def _(_):
                    if self.defocus_checkbox.value:
                        self.frame.defocus = [
                            self.defocus_plane_slider.value,
                            self.defocus_near_far_slider.value,
                            self.defocus_aperture_slider.value,
                        ]
                    else:
                        self.frame.defocus = None
                    self.defocus_plane_slider.visible = self.defocus_checkbox.value
                    self.defocus_near_far_slider.visible = self.defocus_checkbox.value
                    self.defocus_aperture_slider.visible = self.defocus_checkbox.value
                    self.editor.logger.info(f'Defocus of key frame {self._order} set to: {self.frame.defocus}')

                @self.defocus_plane_slider.on_update
                def _(_):
                    if self.frame.defocus is not None:
                        self.frame.defocus[0] = self.defocus_plane_slider.value
                        self.editor.logger.info(f'Defocus plane of key frame {self._order} set to: {self.frame.defocus[0]}')

                @self.defocus_near_far_slider.on_update
                def _(_):
                    if self.frame.defocus is not None:
                        self.frame.defocus[1] = self.defocus_near_far_slider.value
                        self.editor.logger.info(f'Defocus near far of key frame {self._order} set to: {self.frame.defocus[1]}')

                @self.defocus_aperture_slider.on_update
                def _(_):
                    if self.frame.defocus is not None:
                        self.frame.defocus[2] = self.defocus_aperture_slider.value
                        self.editor.logger.info(f'Defocus aperture of key frame {self._order} set to: {self.frame.defocus[2]}')
    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        self._order = value
        self._folder.order = value*2+100
        self._folder.label = f'Key Frame {value}'

    @property
    def json(self):
        return self.frame.json

    def __call__(self, net_g, shape):
        return self.frame(net_g, shape)

class Frame:
    def __init__(self, position: np.ndarray, rotation: np.ndarray, fov: float, exposure: float = None, meta_data: list[torch.Tensor] = None, hdr: bool = None, full_hdr: bool = None, color_temp: float = None, lut: torch.Tensor = None, defocus: list = None):
        self.position = position # [3]
        self.rotation = rotation # [3, 3]
        self.fov = fov
        self.exposure = exposure
        self.meta_data = deepcopy(meta_data)
        for idx, meta_data in enumerate(self.meta_data):
            if not isinstance(meta_data, torch.Tensor):
                self.meta_data[idx] = torch.tensor(meta_data, device=DEVICE)
        self.hdr = hdr
        self.full_hdr = full_hdr
        self.color_temp = color_temp
        self.lut = lut
        self.defocus = defocus

    @property
    def pose(self):
        return np.concatenate([self.rotation, self.position[:, None]], axis=1) # [3, 4]

    @property
    def json(self):
        return {
            'position': self.position.tolist(),
            'rotation': self.rotation.tolist(),
            'fov': self.fov,
            'exposure': self.exposure,
            'meta_data': [meta_data.cpu().numpy().tolist() for meta_data in self.meta_data],
            'hdr': self.hdr,
            'full_hdr': self.full_hdr,
            'color_temp': self.color_temp,
            'lut': self.lut.tolist(),
            'defocus': self.defocus,
        }

    def __call__(self, net_g, shape):
        fov = self.fov / 180.0 * math.pi
        fov_x = fov * (shape[1] / float(shape[0])) / 2
        fov_y = fov / 2
        camera = Camera(colmap_id=0, R=self.rotation, T=self.position, fov_x=fov_x, fov_y=fov_y,
                        image=torch.zeros((3, *shape)), gt_alpha_mask=None,
                        image_name='viser_viewer_fake_img.jpg', uid=0,
                        znear=0.2, zfar=1000,
                        bayer_mask=False).cuda()
        image = render(net_g, camera)['render']

        if self.defocus is not None:
            target_bin = round(self.defocus[0] * 31)
            hists = render_hist(net_g, camera, num_bins=32, near_far=self.defocus[1]).pop('render')
            rendered_pkg = apply_defocus(image, hists, target_bin, 32,
                                         delta_r=1/self.defocus[2],
                                         position=None)
            image = rendered_pkg[0]

        lut = get_lut(self.lut, 16) if self.lut is not None else None
        if self.exposure is not None:
            image = image * 2 ** self.exposure
            image = adjust_color_temperature(image, self.color_temp)

            if self.hdr:
                full_hdr = self.full_hdr if self.full_hdr is not None else False
                image = hdr_plus_photo_finish(image, self.meta_data, lut=lut, full=full_hdr)
            else:
                image = finish_isp(image, self.meta_data, lut=lut, ratio=1)
        else:
            image = apply_lut(image, lut, 16) if lut is not None else image

        return image.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
