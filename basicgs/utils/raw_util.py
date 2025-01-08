import math
import numpy as np
import rawpy
import json
import torch
from torch.nn import functional as F
from kornia.filters import get_gaussian_kernel2d

from .hdr_util import BlendMertens, build_pyramid, pyrup, rgb_to_grayscale


DEVICE = 'cpu'

def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def apply_bayer_mask(im):
    mask = torch.zeros_like(im)
    mask[0, 0::2, 0::2] = 1
    mask[1, 0::2, 1::2] = 1
    mask[1, 1::2, 0::2] = 1
    mask[2, 1::2, 1::2] = 1
    return (im * mask).sum(0, keepdim=True)

def load_metadata(json_file, device=DEVICE):
    meta = read_json(json_file)
    if isinstance(meta, list):
        meta = meta[0]
    wl, bl = meta['WhiteLevel'], meta['BlackLevel']
    wb, ccm2 = meta['AsShotNeutral'], meta['ColorMatrix2']
    shutter = meta['ShutterSpeed']
    noise_profile = meta['NoiseProfile']
    cfa_pattern_dim = meta['CFARepeatPatternDim']
    cfa_pattern = meta['CFAPattern2']

    def convert_to_float(s):
        if '/' in s:
            numerator, denominator = s.split('/')
            return int(numerator) / int(denominator)
        else:
            return float(s)

    wb = [float(v) for v in wb.split(' ')]
    ccm2 = [float(v) for v in ccm2.split(' ')]
    noise_profile = [float(v) for v in noise_profile.split(' ')]
    shutter = convert_to_float(shutter)
    cfa_pattern_dim = [int(v) for v in cfa_pattern_dim.split(' ')]
    cfa_pattern = [int(v) for v in cfa_pattern.split(' ')]

    wl = torch.tensor(float(wl), device=device)
    bl = torch.tensor(float(bl), device=device)
    wb = torch.tensor(wb, device=device)
    ccm2 = torch.tensor(ccm2, device=device).reshape(3, 3)
    shutter = torch.tensor(shutter, device=device)
    cfa_pattern = torch.tensor(cfa_pattern, device=device).reshape(*cfa_pattern_dim)  # [[0, 1], [1, 2]]

    return wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern.float()

def load_raw(raw_file, device=DEVICE):
    with rawpy.imread(raw_file) as raw:
        raw_im = torch.tensor(raw.raw_image_visible.astype('float32'), device=device)
        raw_pattern = torch.tensor(raw.raw_pattern, device=device)

    R = torch.where(raw_pattern==0)
    G1 = torch.where(raw_pattern==1)
    B = torch.where(raw_pattern==2)
    G2 = torch.where(raw_pattern==3)

    out = torch.stack((raw_im[ R[0][0]::2,  R[1][0]::2], #RGBG
                       raw_im[G1[0][0]::2, G1[1][0]::2],
                       raw_im[ B[0][0]::2,  B[1][0]::2],
                       raw_im[G2[0][0]::2, G2[1][0]::2]), axis=0)

    return out


def normalize(raw_im, wl, bl):
    return (raw_im - bl) / (wl - bl)


def half_size_demosaic(bayer_images):
    r = bayer_images[:, 0:1, :, :]
    gr = bayer_images[:, 1:2, :, :]
    b = bayer_images[:, 2:3, :, :]
    gb = bayer_images[:, 3:4, :, :]
    g = (gr + gb) / 2
    linear_rgb = torch.cat([r, g, b], dim=1)
    return linear_rgb


def demosaic(bayer_images):
    """Bilinearly demosaics a batch of RGGB Bayer images."""
    def bilinear_interpolate(x, shape):
        return torch.nn.functional.interpolate(x, shape, mode='bilinear')

    def space_to_depth(x, downscale_factor):
        return torch.nn.functional.pixel_unshuffle(x, downscale_factor)

    def depth_to_space(x, upscale_factor):
        return torch.nn.functional.pixel_shuffle(x, upscale_factor)

    # This implementation exploits how edges are aligned when upsampling with
    # torch.nn.functional.interpolate.

    B, C, H, W = bayer_images.shape
    shape = [H * 2, W * 2]

    red = bayer_images[:, 0:1]
    green_red = bayer_images[:, 1:2]
    blue = bayer_images[:, 2:3]
    green_blue = bayer_images[:, 3:4]

    red = bilinear_interpolate(red, shape)

    green_red = torch.fliplr(green_red)
    green_red = bilinear_interpolate(green_red, shape)
    green_red = torch.fliplr(green_red)
    green_red = space_to_depth(green_red, 2)

    green_blue = torch.flipud(green_blue)
    green_blue = bilinear_interpolate(green_blue, shape)
    green_blue = torch.flipud(green_blue)
    green_blue = space_to_depth(green_blue, 2)

    green_at_red = (green_red[:, 0] + green_blue[:, 0]) / 2
    green_at_green_red = green_red[:, 1]
    green_at_green_blue = green_blue[:, 2]
    green_at_blue = (green_red[:, 3] + green_blue[:, 3]) / 2
    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = depth_to_space(torch.stack(green_planes, dim=1), 2)

    blue = torch.flipud(torch.fliplr(blue))
    blue = bilinear_interpolate(blue, shape)
    blue = torch.flipud(torch.fliplr(blue))

    rgb_images = torch.cat([red, green, blue], dim=1)
    return rgb_images


def cam2rgb_ccm(ccm2):
    rgb2xyz = torch.tensor(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        device=ccm2.device)
    rgb2cam = ccm2 @ rgb2xyz
    rgb2cam /= rgb2cam.sum(axis=-1, keepdims=True)
    cam2rgb = torch.linalg.inv(rgb2cam)
    return cam2rgb


def adjust_exposure(im, p=97):
    """
    blame GPT
    """
    """
    Adjust the exposure of an image using PyTorch such that the white level is set to the p-th percentile.

    :param im: A PyTorch tensor representing the image.
    :param p: Percentile to set the white level (default is 97).
    :return: Adjusted image.
    """
    # Calculate the p-th percentile of the image
    k = 1 + round(0.01 * float(p) * (im.numel() - 1))
    percentile_value = im.view(-1).kthvalue(k).values.item()

    # Avoid division by zero
    if percentile_value == 0:
        return im

    # Adjust the exposure
    adjusted_image = im / percentile_value

    # Clip values to maintain them within the range [0, 1]
    adjusted_image = torch.clamp(adjusted_image, 0, 1)

    return adjusted_image


def gamma_correct(linear, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    srgb0 = 323 / 25 * linear
    srgb1 = (211 * torch.maximum(torch.tensor(eps), linear)**(5 / 12) - 11) / 200
    return torch.where(linear <= 0.0031308, srgb0, srgb1)


def gamma_expansion(srgb, eps=None):
    if eps is None:
        eps = torch.finfo(torch.float32).eps
    linear0 = 25 / 323 * srgb
    linear1 = torch.maximum(torch.tensor(eps), ((200 * srgb + 11) / (211)))**(12 / 5)
    return torch.where(srgb <= 0.04045, linear0, linear1)


def rawnerf_isp(file, half_size=False, no_tone_mapping=False, device=DEVICE):
    if isinstance(file, str):
        im = load_raw(file, device=device)
        wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern = load_metadata(file.replace('.dng', '.json'), device=device)
    elif isinstance(file, tuple) and len(file) == 2:
        im, (wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern) = file
    else:
        raise NotImplementedError

    local_demosaic = half_size_demosaic if half_size else demosaic

    im = normalize(im, wl, bl)
    im = local_demosaic(im.unsqueeze(0)).squeeze(0)
    wb = torch.diag(1 / wb)
    im = torch.einsum('ij,jkl->ikl', cam2rgb_ccm(ccm2) @ wb, im)
    if no_tone_mapping:
        return im
    im = adjust_exposure(im, 97)
    im = gamma_correct(im)
    return im

def rawnerf_isp_with_metadata(file, half_size=False, device=DEVICE):
    if isinstance(file, str):
        im = load_raw(file, device=device)
        wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern = load_metadata(file.replace('.dng', '.json'), device=device)
    elif isinstance(file, tuple) and len(file) == 2:
        im, (wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern) = file
    else:
        raise NotImplementedError

    local_demosaic = half_size_demosaic if half_size else demosaic

    im = normalize(im, wl, bl)
    im = local_demosaic(im.unsqueeze(0)).squeeze(0)
    return im, (wl, bl, wb, ccm2, shutter, noise_profile, cfa_pattern)

def get_ratio(im, meta, *, exp=97):
    _,  _, wb, ccm2 = meta[:4]
    wb = torch.diag(1 / wb)
    im = torch.einsum('ij,jkl->ikl', cam2rgb_ccm(ccm2) @ wb, im)
    if exp > 0:
        im_adj = adjust_exposure(im, exp)
        ratio = im_adj / im
        return ratio[~ratio.isnan()].mean()
    return 1.0

def adjust_color_temperature(image, kelvin):
    # Convert Kelvin to temperature
    temperature = kelvin / 100

    # Initialize RGB scaling factors
    if temperature <= 66:
        red = 255
        green = temperature
        green = 99.4708025861 * np.log(green) - 161.1195681661

        if temperature <= 19:
            blue = 0
        else:
            blue = temperature - 10
            blue = 138.5177312231 * np.log(blue) - 305.0447927307
    else:
        red = temperature - 60
        red = 329.698727446 * ((red) ** -0.1332047592)
        green = temperature - 60
        green = 288.1221695283 * ((green) ** -0.0755148492)
        blue = 255

    # Clamp RGB values to the range [0, 255]
    red = np.clip(red, 0, 255).item()
    green = np.clip(green, 0, 255).item()
    blue = np.clip(blue, 0, 255).item()

    # Calculate scaling factors
    r_scale = red / 255
    g_scale = green / 255
    b_scale = blue / 255

    image[0, :, :] = image[0, :, :] * r_scale
    image[1, :, :] = image[1, :, :] * g_scale
    image[2, :, :] = image[2, :, :] * b_scale
    return image

def apply_lut(im, lut, bit_depth=16):
    im = im * (2 ** bit_depth - 1)
    indices = im.clamp(0, 2 ** bit_depth - 1).long()
    return lut[indices]

def finish_isp(im, meta, *, lut=None, exp=97, ratio=-1):
    _,  _, wb, ccm2 = meta[:4]
    wb = torch.diag(1 / wb)
    im = torch.einsum('ij,jkl->ikl', cam2rgb_ccm(ccm2) @ wb, im)
    if lut is not None:
        im = apply_lut(im, lut)
    if ratio >= 0:
        im = (im * ratio).clip(0, 1)
    elif exp > 0:
        im = adjust_exposure(im, exp)
    else:
        im = im.clip(0, 1)
    im = gamma_correct(im)
    return im

def unsharp_masking_pytorch(img):
    # Create 3-level Gaussian pyramid
    gaussians = build_pyramid(img, 3)

    # Calculate sum of Gaussian blur
    blurred = gaussians[0]  # Use second layer, adjust based on actual effect
    for g in gaussians[1:]:
        g_up = pyrup(g, blurred.shape)
        blurred = blurred + g_up
    blurred /= len(gaussians)

    # Sharpen image: original - blur + original
    lambda_factor = 1.5  # Adjustable enhancement coefficient
    sharp = img + lambda_factor * (img - blurred)

    # Clip results to [0, 1] range
    sharp = torch.clamp(sharp, 0, 1)
    return sharp

def rgb_to_hsv(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Convert an image from RGB to HSV.

    .. image:: _static/img/rgb_to_hsv.png

    The image data is assumed to be in the range of (0, 1).

    Args:
        image: RGB Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.
        eps: scalar to enforce numarical stability.

    Returns:
        HSV version of the image with shape of :math:`(*, 3, H, W)`.
        The H channel values are in the range 0..2pi. S and V are in the range 0..1.

    .. note::
       See a working example `here <https://kornia.github.io/tutorials/nbs/color_conversions.html>`__.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_hsv(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    max_rgb, argmax_rgb = image.max(-3)
    min_rgb, argmin_rgb = image.min(-3)
    deltac = max_rgb - min_rgb

    v = max_rgb
    s = deltac / (max_rgb + eps)

    deltac = torch.where(deltac == 0, torch.ones_like(deltac), deltac)
    rc, gc, bc = torch.unbind((max_rgb.unsqueeze(-3) - image), dim=-3)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * deltac
    h3 = (gc - rc) + 4.0 * deltac

    h = torch.stack((h1, h2, h3), dim=-3) / deltac.unsqueeze(-3)
    h = torch.gather(h, dim=-3, index=argmax_rgb.unsqueeze(-3)).squeeze(-3)
    h = (h / 6.0) % 1.0
    h = 2.0 * math.pi * h  # we return 0/2pi output

    return torch.stack((h, s, v), dim=-3)

def hsv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an image from HSV to RGB.

    The H channel values are assumed to be in the range 0..2pi. S and V are in the range 0..1.

    Args:
        image: HSV Image to be converted to HSV with shape of :math:`(*, 3, H, W)`.

    Returns:
        RGB version of the image with shape of :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = hsv_to_rgb(input)  # 2x3x4x5

    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    h: torch.Tensor = image[..., 0, :, :] / (2 * math.pi)
    s: torch.Tensor = image[..., 1, :, :]
    v: torch.Tensor = image[..., 2, :, :]

    hi: torch.Tensor = torch.floor(h * 6) % 6
    f: torch.Tensor = ((h * 6) % 6) - hi
    one: torch.Tensor = torch.tensor(1.0, device=image.device, dtype=image.dtype)
    p: torch.Tensor = v * (one - s)
    q: torch.Tensor = v * (one - f * s)
    t: torch.Tensor = v * (one - (one - f) * s)

    hi = hi.long()
    indices: torch.Tensor = torch.stack([hi, hi + 6, hi + 12], dim=-3)
    out = torch.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q), dim=-3)
    out = torch.gather(out, -3, indices)

    return out

def hdr_plus_photo_finish(im, meta, full=False, *, lut=None):
    _,  _, wb, ccm2 = meta[:4]

    ## white balance and color correction
    wb = torch.diag(1 / wb)
    im = torch.einsum('ij,jkl->ikl', cam2rgb_ccm(ccm2) @ wb, im)

    ## lut
    if lut is not None:
        im = apply_lut(im, lut)

    ## exposure fusion and gamma correction
    # TBFix
    evn = gamma_correct((im / 4).clip(0, 1).unsqueeze(0))
    ev0 = gamma_correct(im.clip(0, 1).unsqueeze(0))
    evp = gamma_correct((im * 4).clip(0, 1).unsqueeze(0))
    blend_op = BlendMertens(1.0, 1.0, 1.0, clip=True)
    im  = blend_op(evn, ev0, evp).squeeze(0)
    im = gamma_expansion(im)  # 3, H, W

    ## Global tone adjustment
    im = im / (1 + rgb_to_grayscale(im))  # reinhard tonemapping
    im = gamma_correct(im)
    im = 3 * im ** 2 - 2 * im ** 3

    if full:
        ## Sharpening
        im = unsharp_masking_pytorch(im.unsqueeze(0))

        ## "dehazing"
        pixel_count = im.numel()
        low_threshold = 1.0 * 0.07  # 1.0 is the white level
        clamped_count = int(pixel_count * 0.001)
        flattened_tensor = im.flatten()
        clamp_threshold = torch.kthvalue(flattened_tensor, clamped_count + 1).values
        # Pixels below threshold are set to zero
        im[torch.logical_and(im < clamp_threshold, im < low_threshold)] = 0

        ## Hue-specific color adjustments
        hsv = rgb_to_hsv(im).squeeze()
        # Hue conversion parameters
        bluish_cyan_range = (80 / 360, 100 / 360)
        purple_range = (140 / 360, 160 / 360)
        light_blue_hue = 90 / 360
        # Saturation enhancement parameters
        blue_range = (100 / 360, 130 / 360)
        green_range = (50 / 360, 80 / 360)
        saturation_increase = 1.3
        # Hue conversion logic
        mask_bluish_cyan = (hsv[0, :, :] >= bluish_cyan_range[0]) & (hsv[0, :, :] <= bluish_cyan_range[1])
        mask_purple = (hsv[0, :, :] >= purple_range[0]) & (hsv[0, :, :] <= purple_range[1])
        hsv[0, mask_bluish_cyan | mask_purple] = light_blue_hue
        # Saturation adjustment logic
        mask_blue = (hsv[0, :, :] >= blue_range[0]) & (hsv[0, :, :] <= blue_range[1])
        mask_green = (hsv[0, :, :] >= green_range[0]) & (hsv[0, :, :] <= green_range[1])
        hsv[1, mask_blue | mask_green] *= saturation_increase
        hsv[1, :, :] = torch.clamp(hsv[1, :, :], 0, 1)  # Ensure saturation is within valid range
        # Convert back to RGB color space
        im = hsv_to_rgb(hsv.unsqueeze(0)).squeeze()

    return im

def apply_defocus(rendered_image, hists, target_bin,
                  num_bins=32, delta_r=None, delta_d=[0.0, 0.0], position=None):
    """
    # TODO: Render translation
    """

    def _pad_to_size(img, target_h, target_w):
        """
        将图像填充到指定的尺寸(target_h, target_w)。
        """
        h, w = img.shape[-2:]
        pad_vert = target_h - h
        pad_horiz = target_w - w
        pad_top = pad_vert // 2
        pad_bottom = pad_vert - pad_top
        pad_left = pad_horiz // 2
        pad_right = pad_horiz - pad_left
        return F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    def _center_crop(img, orig_h, orig_w):
        """
        从填充后的图像中心裁剪出原始尺寸(orig_h, orig_w)的图像。
        """
        h, w = img.shape[-2:]
        start_h = (h - orig_h) // 2
        start_w = (w - orig_w) // 2
        end_h = start_h + orig_h
        end_w = start_w + orig_w
        return img[..., start_h:end_h, start_w:end_w]

    def _get_gaussian_kernels_freq(kernel_sizes, H, W):
        kernels = []
        max_kernel_size = max(kernel_sizes)
        H_prime, W_prime = H + max_kernel_size - 1, W + max_kernel_size - 1
        for kernel_size in kernel_sizes:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
            kernel = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma), device='cuda')
            kernel = _pad_to_size(kernel, max_kernel_size, max_kernel_size)
            kernel_freq = torch.fft.fft2(kernel, s=(H_prime, W_prime))
            kernels.append(kernel_freq)
        return torch.stack(kernels, dim=0) # bin, 1, H, W

    def _apply_blur_in_freq(blur_kernels, image):
        H_prime, W_prime = blur_kernels.shape[2:]
        H, W = image.shape[-2:]
        image = _pad_to_size(image, H_prime, W_prime)
        image = torch.fft.fft2(image)
        result = image * blur_kernels
        return torch.fft.ifft2(result).real[:, :, -H:, -W:]
        # return _center_crop(torch.fft.ifft2(result).real, H, W)

    def _apply_translate(translation, image):
        # image: B, C, H, W
        # translation: B, 3, 2
        grid = F.affine_grid(translation, image.size(), align_corners=False)
        translated_image = F.grid_sample(image, grid, align_corners=False)
        return translated_image

    def _apply_defocus(mpis, hists, kernel_sizes, translations):
        H, W = mpis.shape[-2:]
        kernels = _get_gaussian_kernels_freq(kernel_sizes, H, W)
        mpis_after_blur = _apply_blur_in_freq(kernels, mpis)
        hists_after_blur = _apply_blur_in_freq(kernels, hists)

        mpis_after_trans = _apply_translate(translations, mpis_after_blur)
        hists_after_trans = _apply_translate(translations, hists_after_blur)

        result = 0
        for idx in reversed(range(len(kernel_sizes))):
            result = mpis_after_trans[idx] + result * (1 - hists_after_trans[idx])
        return result

    if position is not None:
        x, y = position
        target_hist = hists[:, y, x]
        target_bin  = target_hist.argmax().item()
    color = rendered_image # 3, H, W

    # MPI
    hists = hists.unsqueeze(1) # bin, 1, H, W
    color = color.unsqueeze(0)
    mpi_rep = hists * color # bin, 3, H, W

    # get defocus data
    bins = torch.linspace(0, num_bins - 1, num_bins, device=hists.device)
    kernel_size = torch.floor(delta_r * torch.abs(target_bin - bins)).long() # bins
    kernel_size = kernel_size * 2 + 1
    translation = torch.tensor([delta_d], device=hists.device) * (num_bins - bins.unsqueeze(1)) # bins, 2
    translation = translation.unsqueeze(-1) # bins, 2, 1
    padded_translation = torch.eye(2, device=hists.device).unsqueeze(0).repeat(num_bins, 1, 1) # bins, 2, 2
    translation = torch.cat((padded_translation, translation), dim=-1) # bins, 2, 3
    render = _apply_defocus(mpi_rep, hists, kernel_size, translation)
    return render, target_bin / float(num_bins-1)
