import torch
from torch import nn
from basicgs.gaussians.util.render import render_final_opacity, render_depth_raywise, render_hist, render_near_far
from basicgs.utils.registry import LOSS_REGISTRY
from .loss_util import l1_loss, mse_loss, charbonnier_loss, _reduction_modes, weight_reduce_loss


@LOSS_REGISTRY.register()
class TLogReg(nn.Module):
    def __init__(self, net_g, loss_weight=1.0, eps=1e-3, reduction='mean'):
        super(TLogReg, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.name = 'tlog'
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, net_g, camera, render_pkg, current_iter, weight=None, **kwargs):
        T = render_final_opacity(net_g, camera)['render']
        T = torch.clamp(T + self.eps, max=1.0)
        loss = -torch.log(T)
        return self.loss_weight * weight_reduce_loss(loss, weight, reduction=self.reduction)

def dist_loss(near, far, ws, inter_weight=1.0, intra_weight=1.0, eps=torch.finfo(torch.float32).eps):
    g = lambda x : 1 / x
    bins = ws.size(1)
    t = torch.linspace(near+eps, far, bins+1, device=ws.device)  # same naming as multinerf
    s = (g(t) - g(near+eps)) / (g(far) - g(near+eps))            # convert t to s
    us = (s[1:] + s[:-1]) / 2
    dus = torch.abs(us[:, None] - us[None, :])
    loss_inter = torch.sum(ws * torch.sum(ws[..., None, :] * dus[None, ...], dim=-1), dim=-1)
    ds = s[1:] - s[:-1]
    loss_intra = torch.sum(ws**2 * ds[None, :], dim=-1) / 3
    return loss_inter * inter_weight + loss_intra * intra_weight

@LOSS_REGISTRY.register()
class DistortionReg(nn.Module):
    def __init__(self, net_g, bins, shape, inter_weight, intra_weight,
                 loss_weight=1.0, near_far=(0.2, 1000.0), reduction='mean'):
        super(DistortionReg, self).__init__()
        self.name = 'dist'

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.shape = shape
        self.bins = bins
        self.near_far = near_far
        self.inter_weight = inter_weight
        self.intra_weight = intra_weight

    def forward(self, net_g, camera, render_pkg, current_iter, weight=None):
        with torch.no_grad():
            xx, yy, zz, pixel_filter = render_depth_raywise(net_g, camera, shape=self.shape, return_filter=True, return_all=True)
            visible_zz = zz[pixel_filter]
            near = max(self.near_far[0], visible_zz.min().item())
            far  = min(self.near_far[1], visible_zz.max().item())

        out_pkg = render_hist(net_g, camera, self.bins, self.shape, (near, far))

        hist = out_pkg['render']
        curr_rays = hist.permute(1, 2, 0).reshape(-1, self.bins)
        l_dist = dist_loss(near, far, curr_rays,
                           inter_weight=self.inter_weight, intra_weight=self.intra_weight)
        l_dist = weight_reduce_loss(l_dist, weight=weight, reduction=self.reduction)
        return l_dist * self.loss_weight

@LOSS_REGISTRY.register()
class NearFarReg(nn.Module):
    def __init__(self, net_g, loss_weight=1.0, loss_func='l1_loss', shape=None, reduction='mean'):
        super(NearFarReg, self).__init__()
        self.name = 'nearfar'
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_func = eval(loss_func)
        self.shape = shape

    def forward(self, net_g, camera, render_pkg, current_iter, weight=None):
        assert 'near_indexes' in render_pkg and 'far_indexes' in render_pkg
        near_far_indexes = render_pkg['near_indexes'].detach(), render_pkg['far_indexes'].detach()

        near_far_pack = render_near_far(net_g, camera, near_far_indexes, shape=self.shape)
        near   = near_far_pack['near']
        far    = near_far_pack['far']
        near_T = near_far_pack['near_final_opacity']
        far_T  = near_far_pack['far_final_opacity']
        near   = near / near_T
        far    = far / far_T
        mask = ~((torch.isnan(near) | torch.isinf(near)) | (torch.isnan(far) | torch.isinf(far)))
        l_near_far_reg = self.loss_func(near[mask], far[mask]) * near_T[mask] * far_T[mask]
        l_near_far_reg = weight_reduce_loss(l_near_far_reg, weight, reduction=self.reduction)
        return self.loss_weight * l_near_far_reg
