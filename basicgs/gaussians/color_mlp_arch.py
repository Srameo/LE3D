import torch
from torch import nn

from basicgs.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class TinyColorMLP(nn.Module):
    def __init__(self, in_feats=16, dir_feats=3,
                 mid_feats_list=[16, 16], out_feats=3,
                 final_bias=None, final_act='torch.exp',
                 act='leaky_relu',detach=False) -> None:
        super().__init__()
        self.detach = detach
        if act == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError
        assert dir_feats == 3 or dir_feats == 9 or dir_feats == 19
        self.dir_feats = dir_feats
        feats = [in_feats+dir_feats] + mid_feats_list + [out_feats]
        self.linears = nn.ModuleList([nn.Linear(feats[i], feats[i+1]) for i in range(len(feats)-2)])
        self.linears.append(nn.Linear(feats[-2], feats[-1], bias=False)) # no bias for the last linear
        self.final_act = eval(final_act)
        if final_bias is not None:
            self.final_bias = nn.Parameter(torch.ones(1, out_feats) * final_bias)
        else:
            self.final_bias = torch.zeros(1, out_feats)

    def forward(self, color_feats, dirs, w_bias=True):
        if self.detach:
            dirs = dirs.detach()
        if self.dir_feats >= 9:
            x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            dirs = torch.cat([dirs, xx, yy, zz, xy, yz, xz], dim=-1)
            if self.dir_feats == 19:
                xxx, xxy, xyy = xx * x, xx * y, xy * y
                zzz, xzz, xxz = zz * z, xz * z, xx * z
                yyy, yzz, yyz = yy * y, yz * z, yy * z
                xyz = xy * z
                dirs = torch.cat([dirs, xxx, xxy, xyy,
                                        zzz, xzz, xxz,
                                        yyy, yzz, yyz,
                                        xyz
                                 ], dim=-1)
        final_bias = self.final_bias
        if isinstance(color_feats, tuple):
            final_bias = color_feats[0]
            color_feats = color_feats[1]
        out = torch.cat([color_feats, dirs], dim=-1)
        for linear in self.linears[:-1]:
            out = self.act(linear(out))
        return self.final_act(self.linears[-1](out) + \
                              (final_bias if w_bias else 0)) # with bias or without bias
