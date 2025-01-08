import os
from os import path as osp
import subprocess
import cv2
import imageio_ffmpeg as iio_ffmpeg
import numpy as np

def RGB_to_YUV(RGB, gamut="bt2020", bits=10, video_range="full", formation="420"):
    if RGB.dtype == "uint8":
        RGB = RGB / 255.0
    height, width = RGB.shape[:2]

    if bits == 8:
        dtype = np.uint8
    else:
        dtype = np.uint16

    if gamut == "bt709":
        YCbCr = RGB709_to_YCbCr709(RGB)
    elif gamut == "bt2020":
        YCbCr = RGB2020_to_YCbCr2020(RGB)
    else:
        raise Exception("gamut param error!")

    Y = YCbCr[..., 0]
    Cb = YCbCr[..., 1]
    Cr = YCbCr[..., 2]

    if video_range == "limited":
        D_Y = np.clip(np.round(Y * 219 + 16), 16, 235).astype(dtype) * np.power(2, bits - 8)
        D_Cb = np.clip(np.round(Cb * 224 + 128), 16, 240).astype(dtype) * np.power(2, bits - 8)
        D_Cr = np.clip(np.round(Cr * 224 + 128), 16, 240).astype(dtype) * np.power(2, bits - 8)

    elif video_range == "full":
        D_Y = np.clip(np.round(Y * 255), 0, 255).astype(dtype) * np.power(2, bits - 8)
        D_Cb = np.clip(np.round(Cb * 254 + 128), 1, 255).astype(dtype) * np.power(2, bits - 8)
        D_Cr = np.clip(np.round(Cr * 254 + 128), 1, 255).astype(dtype) * np.power(2, bits - 8)

    else:
        raise Exception("param: video_range error!")

    y_size = height * width
    uv_size = height // 2 * width // 2
    frame_len = y_size * 3 // 2

    if formation == "420":
        U = cv2.resize(D_Cb, None, None, 0.5, 0.5).flatten()
        V = cv2.resize(D_Cr, None, None, 0.5, 0.5).flatten()

        yuv = np.empty(frame_len, dtype=dtype)
        yuv[:y_size] = D_Y.flatten()
        yuv[y_size: y_size + uv_size] = U
        yuv[y_size + uv_size:] = V
        return yuv

    elif formation == "444":
        Y = D_Y
        U = D_Cb
        V = D_Cr
        return cv2.merge((Y, U, V))


def RGB709_to_YCbCr709(RGB):
    if RGB.dtype == "uint8":
        RGB = RGB / 255.0

    m_RGB709_to_YCbCr709 = np.array([[0.21260000, 0.71520000, 0.07220000],
                                     [-0.11457211, -0.38542789, 0.50000000],
                                     [0.50000000, -0.45415291, -0.04584709]])

    return np.matmul(RGB, m_RGB709_to_YCbCr709.T)


def RGB2020_to_YCbCr2020(RGB):
    m_RGB2020_to_YCbCr2020 = np.array([[0.26270000, 0.67800000, 0.05930000],
                                       [-0.13963006, -0.36036994, 0.50000000],
                                       [0.50000000, -0.45978570, -0.04021430]])

    return np.matmul(RGB, m_RGB2020_to_YCbCr2020.T)

class Exporter:
    def __init__(self, file_path, fps, resolution):
        self.file_path = file_path
        os.makedirs(osp.dirname(file_path), exist_ok=True)
        self.fps = fps
        self.index = 0
        self.resolution = resolution  # (H, W)
        self.writer = subprocess.Popen(
            [
                iio_ffmpeg.get_ffmpeg_exe(),
                "-y",  # 覆盖输出文件
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p10le",  # 输入为 YUV420p10le
                "-s", f"{int(self.resolution[1])}x{int(self.resolution[0])}",  # 宽x高
                "-r", str(self.fps),  # 帧率
                "-i", "-",  # 从 stdin 读取
                "-an",  # 无音频
                "-c:v", "libx265",  # 使用 H.265 编码
                "-tag:v", "hvc1",
                "-preset", "slow",
                "-crf", "18",
                "-pix_fmt", "yuv420p10le",  # 输出为 YUV420p10le
                # HDR10 相关参数
                "-color_primaries", "bt2020",
                "-colorspace", "bt2020nc",
                "-color_trc", "arib-std-b67",
                "-x265-params", (
                    "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc:range=full"
                ),
                self.file_path,
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,  # 捕获错误输出
        )

    def quantize_frame(self, frame, bits=10):
        # 确保帧数据在 [0, 1] 范围内
        frame = np.clip(frame, 0.0, 1.0)
        # 转换为 16 位数据 (HDR 需要高精度)
        frame = (frame * (2**bits - 1)).astype(np.uint16)
        return frame

    def add_frame(self, frame):
        # quantized_frame = self.quantize_frame(frame)
        # yuv_frame = RGB_to_YUV(quantized_frame, bits=10)
        yuv_frame = RGB_to_YUV((frame), bits=10)
        frame_bytes = yuv_frame.tobytes()
        self.writer.stdin.write(frame_bytes)
        print(f"Added frame {self.index}")
        # writeYUVFile(f"output_{self.index}.yuv", frame_bytes)
        # exit()
        self.index += 1

    def close(self):
        self.writer.stdin.close()
        self.writer.wait()  # 等待进程结束
        stderr = self.writer.stderr.read().decode()
        if self.writer.returncode != 0:
            print(f"ffmpeg error: {stderr}")
        else:
            print(f"Video saved to {self.file_path}")
