from os import path as osp
import time

import numpy as np
import torch
from basicgs.viewers.utils import get_lut
from basicgs.gaussians.util.render import render_hist
from basicgs.viewers.base_viewer import BaseViewer
from basicgs.utils.raw_util import adjust_color_temperature, load_metadata, finish_isp, hdr_plus_photo_finish, apply_defocus
from basicgs.utils.registry import VIEWER_REGISTRY

DEVICE = 'cuda'

@VIEWER_REGISTRY.register()
class Le3dViewer(BaseViewer):
    def __init__(self, exp_opt, net_g, log_dir, update_freq=None, total_iters=None, port=8097) -> None:
        exp_root = exp_opt['path']['experiments_root']
        meta_path = osp.join(exp_root, 'meta_data.json')
        with torch.no_grad():
            self.meta_data = list(load_metadata(meta_path, DEVICE))
            self.wb_backup = self.meta_data[2].cpu().numpy()
        super().__init__(exp_opt, net_g, log_dir, update_freq, total_iters, port)

    def init_components(self):
        super().init_components()
        with self.post_processing_tab:
            with self.viser_server.gui.add_folder('ISP', order=0):
                self.exposure_value_slider = self.viser_server.gui.add_slider('Exposure Value', -8, 8, 0.01, 0.0)
                self.isp_enable_checkbox = self.viser_server.gui.add_checkbox('Enable ISP', True)
                self.wb_value_array = self.viser_server.gui.add_vector3('White Balance', self.wb_backup, disabled=not self.isp_enable_checkbox.value)
                self.color_temp_slider = self.viser_server.gui.add_slider('Color Temperature', min=2000, max=40000, step=100, initial_value=6500)
                self.hdr_checkbox = self.viser_server.gui.add_checkbox('HDR', False)
                self.full_hdr_checkbox = self.viser_server.gui.add_checkbox('Full HDR', False, visible=self.hdr_checkbox.value)

        @self.isp_enable_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.wb_value_array.disabled = not self.isp_enable_checkbox.value
                self.color_temp_slider.disabled = not self.isp_enable_checkbox.value

        @self.hdr_checkbox.on_update
        def _(_):
            with self.user_interactive_mode:
                self.full_hdr_checkbox.visible = self.hdr_checkbox.value

    @torch.no_grad()
    def get_current_image(self, camera, require_other_process=False):
        rendered_image, render_time, snapshot = super().get_current_image(camera, require_other_process=True)
        snapshot['isp_enable'] = self.isp_enable_checkbox.value
        if self.defocus_checkbox.value:
            target_bin = round(self.defocus_plane_slider.value * 31)
            hists = render_hist(self.net_g, camera, num_bins=32, near_far=self.defocus_near_far_slider.value).pop('render')
            rendered_pkg = apply_defocus(rendered_image, hists, target_bin, 32,
                                         delta_r=1/self.defocus_aperture_slider.value,
                                         position=self.current_click_pos)
            rendered_image = rendered_pkg[0]
            if self.current_click_pos is not None:
                self.defocus_plane_slider.value = rendered_pkg[1]
        rendered_image = (rendered_image * 2 ** self.exposure_value_slider.value)
        rendered_image = adjust_color_temperature(rendered_image, self.color_temp_slider.value)
        if snapshot['isp_enable'] and snapshot['render_image']:
            start_time = time.time()
            self.meta_data[2] = torch.tensor(self.wb_value_array.value, device=DEVICE)
            lut = get_lut(self.get_lut_control_points(), 16) if self.apply_lut_checkbox.value else None
            if self.hdr_checkbox.value:
                rendered_image = hdr_plus_photo_finish(rendered_image, self.meta_data, lut=lut, full=self.full_hdr_checkbox.value)
            else:
                rendered_image = finish_isp(rendered_image, self.meta_data, lut=lut, ratio=1)
            torch.cuda.synchronize()
            isp_time = time.time() - start_time
            render_time = render_time + isp_time
        if not require_other_process:
            rendered_image = rendered_image.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        return rendered_image, render_time, snapshot
