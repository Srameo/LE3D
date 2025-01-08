import os
import random
import sys
import numpy as np
import torch
from torch.utils import data
from torch.nn import functional as F
from basicgs.utils.registry import DATASET_REGISTRY
from basicgs.data.camera_utils import Camera
from basicgs.data.colmap_utils import read_extrinsics_binary, read_extrinsics_text, read_intrinsics_binary, read_intrinsics_text, qvec2rotmat, get_nerf_pp_norm
from basicgs.data.colmap_utils import read_points3D_binary, read_points3D_text
from basicgs.gaussians.util import focal2fov, store_ply
from basicgs.utils.raw_util import load_raw, load_metadata, half_size_demosaic, demosaic, normalize
from imageio import imread


@DATASET_REGISTRY.register()
class ColmapDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.scene_root = opt['scene_root']
        self.image_dir = opt['image_dir']
        self.image_suffix = opt['image_suffix']
        self.meta_dir = opt.get('meta_dir', None)
        self.meta_suffix = opt.get('meta_suffix', 'json')
        self.apply_bayer_mask = opt.get('apply_bayer_mask', False)

        self.split = opt.get('split', 'train')

        self.half_size_demosaic = opt.get('half_size_demosaic', False)
        self.llffhold = opt.get('llffhold', None)
        self.test_cams = opt.get('test_cams', [])
        self.exclude_cams = opt.get('exclude_cams', [])
        self.downsample = opt.get('downsample', 1)
        self.downsample_mode = opt.get('downsample_mode', 'bicubic')
        self.max_width = opt.get('max_width', 1600)

        self.load_colmap()
        self.load_camera_infos()
        self.load_cameras()

    def load_colmap(self):
        try:
            cameras_extrinsic_file = os.path.join(self.scene_root, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.scene_root, "sparse/0", "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.scene_root, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.scene_root, "sparse/0", "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

        self.cam_extrinsics = cam_extrinsics
        self.cam_intrinsics = cam_intrinsics

        ply_path = os.path.join(self.scene_root, "sparse/0/points3D.ply")
        bin_path = os.path.join(self.scene_root, "sparse/0/points3D.bin")
        txt_path = os.path.join(self.scene_root, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            store_ply(ply_path, xyz, rgb)
        self.ply_path = ply_path

    def _im_read(self, image_name):
        image_path = '.'.join([image_name, self.image_suffix])
        if self.meta_dir is not None:
            meta_path = '.'.join(
                [image_name.replace(f'{os.path.sep}{self.image_dir}', f'{os.path.sep}{self.meta_dir}'), self.meta_suffix])
            meta = load_metadata(meta_path)
        else:
            meta = None
        if self.image_suffix in ['npy', 'NPY']:
            npy = np.load(image_path)
            if npy.shape[0] != 3:
                npy = npy.transpose(2, 0, 1)
            image = torch.from_numpy(npy).float()
        elif self.image_suffix in ['dng', 'DNG']:
            assert self.meta_dir is not None, 'meta_dir is not specified for dng format'
            wl, bl = meta[:2]
            image = load_raw(image_path, 'cuda')
            image = normalize(image, wl, bl)
            if self.half_size_demosaic:
                image = half_size_demosaic(image.unsqueeze(0))[0].cpu()
            else:
                image = demosaic(image.unsqueeze(0))[0].cpu()
        elif self.image_suffix in ['jpg', 'JPG', 'png', 'PNG']:
            image = torch.from_numpy(imread(image_path).transpose(2, 0, 1)).float() / 255.0
        else:
            raise NotImplementedError
        return image, meta

    def load_camera_infos(self):

        cam_infos = []
        # load cameras from intr or extr
        count = 0
        for idx, key in enumerate(self.cam_extrinsics):
            if self.llffhold is not None:
                if self.split == 'train' and idx % self.llffhold == 0:
                    continue
                elif self.split == 'val' and idx % self.llffhold != 0:
                    continue
            sys.stdout.write('\r')
            # the exact output you're looking for:
            sys.stdout.write("Reading camera {}/{}".format(count+1, len(self.cam_extrinsics)))
            sys.stdout.flush()
            count += 1

            extr = self.cam_extrinsics[key]
            intr = self.cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)

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

            image_path = os.path.join(self.scene_root, self.image_dir, os.path.basename(extr.name))
            # change name to dng
            image_name = os.path.splitext(image_path)[0]
            image_key = os.path.basename(image_name)
            if self.llffhold is None and image_key in self.test_cams and self.split == 'train':
                continue
            elif self.llffhold is None and image_key not in self.test_cams and self.split == 'val':
                continue
            if image_key in self.exclude_cams:
                print(f'Exclude {image_name}')
                continue

            image, meta = self._im_read(image_name)

            cam_info = {
                'colmap_id': uid,
                'R': R, 'T': T,
                'fov_x': fov_x, 'fov_y': fov_y,
                'image': image, 'meta_data': meta,
                'image_name': image_name, 'image_path': image_path,
                'width': width, 'height': height
            }
            cam_infos.append(cam_info)
        cam_infos = sorted(cam_infos.copy(), key = lambda x : x['image_name'])
        self.cam_infos = cam_infos
        sys.stdout.write('\n')

    def load_cameras(self):
        cam_infos = self.cam_infos

        cameras = []
        for uid, cam_info in enumerate(cam_infos):
            cam_info['uid'] = uid
            cam_info['gt_alpha_mask'] = None
            image = cam_info['image']
            _, h, w = image.shape
            if w < self.max_width and self.downsample == 1:
                new_w = w
            elif self.downsample > 1:
                if w / self.downsample > self.max_width:
                    new_w = self.max_width
                else:
                    new_w = w / self.downsample
            else:
                new_w = self.max_width

            new_h = int(h * new_w / w)
            if new_w != w:
                if self.downsample_mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
                    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode=self.downsample_mode, align_corners=False).squeeze(0)
                else:
                    image = F.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode=self.downsample_mode).squeeze(0)

            cam_info['image'] = image
            cameras.append(Camera(bayer_mask=self.apply_bayer_mask, **cam_info))

        self.cams = cameras

        self.nerf_normalization = get_nerf_pp_norm(self.cams)

        import json
        if self.split == 'train':
            fp = os.path.join(self.scene_root, f'{os.path.basename(self.scene_root)}_cameras.json')
            with open(fp, 'w') as f:
                cameras_json = {}
                for camera in cameras:
                    from basicgs.gaussians.util import get_world_to_view_wts
                    camera_json = {'K': [[camera.focal_x, 0, camera.image_width // 2, 0],
                                         [0, camera.focal_y, camera.image_height // 2, 0],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]],
                                   'W2C': get_world_to_view_wts(camera.R, camera.T).tolist(),
                                   'img_size': [camera.image_width, camera.image_height]}
                    cameras_json[camera.image_name] = camera_json
                json.dump(cameras_json, f, indent=4)
            print(f"{fp} saved.")

    @property
    def shutter_set(self):
        if not hasattr(self, '_max_shutter'):
            shutter_set = []
            assert self.meta_dir is not None, 'Only RAW image based training supports shutter set'
            mshutter = -1
            for cam in self.cams:
                if cam.meta_data[4] > mshutter:
                    mshutter = cam.meta_data[4]
                shutter_set.append(cam.meta_data[4].item())
            self._max_shutter = mshutter
            self._shutter_set = list(sorted(list(set(shutter_set))))
        return self._max_shutter, self._shutter_set

    def __len__(self):
        # 返回数据集的大小
        return len(self.cams)

    def shuffle(self):
        # 打乱数据集
        random.shuffle(self.cams)

    def __getitem__(self, idx):
        additional_data = {}
        cam = self.cams[idx]
        if self.meta_dir is not None:
            max_shutter, shutter_set = self.shutter_set
            additional_data['max_shutter'] = max_shutter
            additional_data['shutter_set'] = shutter_set
            additional_data['shutter'] = cam.meta_data[4].item()
            additional_data['ratio'] = additional_data['shutter'] / max_shutter
        additional_data['cameras_count'] = len(self)
        # 返回第 idx 个样本
        return {
            'camera': cam,
            'cameras_extent': torch.tensor(self.nerf_normalization['radius']).float(),
            'cameras_center': -torch.tensor(self.nerf_normalization['translate']).float(),
            **additional_data
        }
