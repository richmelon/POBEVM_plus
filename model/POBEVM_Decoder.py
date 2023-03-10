import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Optional

class SOBE(nn.Module):
    def __init__(self, feature_channels):  # feature_channels就是backbone输出的各个特征的通道数，
        super().__init__()
        self.refiner3 = SOBE_Block(feature_channels[3],  feature_channels[2])
        self.refiner2 = SOBE_Block(feature_channels[2],  feature_channels[1])
        self.refiner1 = SOBE_Block(feature_channels[1],  feature_channels[0])
        self.decoder_fgr = OutputBlock(feature_channels[0], 3, feature_channels[0])

    def forward_time_series(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, p4: Tensor, r1: Optional[int] = None, r2: Optional[int] = None, r3: Optional[int] = None,
               r4: Optional[int] = None
               ):
        '''

        Args:
            x1: low level feature
            f2: high level feature
            p3/p2/p1: prediction

        Returns:

        '''
        B, T, C, H, W = f1.shape
        f1 = f1.flatten(0, 1)
        f2 = f2.flatten(0, 1)
        f3 = f3.flatten(0, 1)
        f4 = f4.flatten(0, 1)
        p4 = p4.flatten(0, 1)


        x3, p3, r1 = self.refiner3(f3, f4, p4, r1)
        x2, p2, r2 = self.refiner2(f2, x3, p3, r2)
        x1, p1, r3 = self.refiner1(f1, x2, p2, r3)

        p3 = p3.unflatten(0, (B,T))
        p2 = p2.unflatten(0, (B, T))
        p1 = p1.unflatten(0, (B, T))
        x1 = x1.unflatten(0, (B, T))

        fgr_feature, p0, fgr = self.decoder_fgr(x1, s0)
        return fgr_feature, fgr, p0, p1, p2, p3

    def forward_single_frame(self,
                s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, p4: Tensor, r1: Optional[int] = None, r2: Optional[int] = None, r3: Optional[int] = None,
               r4: Optional[int] = None):
        '''

        Args:
            x1: low level feature
            f2: high level feature
            p3/p2/p1: prediction

        Returns:

        '''
        x3, prediction_mat3, r1 = self.refiner3(f3, f4, p4, r1)
        x2, prediction_mat2, r2 = self.refiner2(f2, x3, prediction_mat3, r2)
        x1, prediction_mat1, r3 = self.refiner1(f1, x2, prediction_mat2, r3)
        fgr_feature, pha, fgr = self.decoder_fgr(x1, s0)

        return fgr_feature, fgr, pha, prediction_mat1, prediction_mat2, prediction_mat3

    def forward(self, s0: Tensor, f1: Tensor, f2: Tensor, f3: Tensor, f4: Tensor, p4: Tensor, r1: Optional[int] = None, r2: Optional[int] = None, r3: Optional[int] = None,
               r4: Optional[int] = None):
        if s0.ndim == 5:
            return self.forward_time_series(s0, f1, f2, f3, f4, p4, r1, r2, r3, r4)
        else:
            return self.forward_single_frame(s0, f1, f2, f3, f4, p4, r1, r2, r3, r4)


class ConvGRU(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 3,
                 padding: int = 1):
        super().__init__()
        self.channels = channels
        self.ih = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size, padding=padding),
            nn.Sigmoid()
        )
        self.hh = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size, padding=padding),
            nn.Tanh()
        )

    def forward_single_frame(self, x, h):
        r, z = self.ih(torch.cat([x, h], dim=1)).split(self.channels, dim=1)
        c = self.hh(torch.cat([x, r * h], dim=1))
        h = (1 - z) * h + z * c
        return h, h

    def forward_time_series(self, x, h):
        o = []
        for xt in x.unbind(dim=1):
            ot, h = self.forward_single_frame(xt, h)
            o.append(ot)
        o = torch.stack(o, dim=1)
        return o, h

    def forward(self, x, h: Optional[Tensor]):
        if h is None:
            h = torch.zeros((x.size(0), x.size(-3), x.size(-2), x.size(-1)),
                            device=x.device, dtype=x.dtype)

        if x.ndim == 5:
            return self.forward_time_series(x, h)
        else:
            return self.forward_single_frame(x, h)

class SOBE_Block(nn.Module):
    def __init__(self, channel1, channel2):
        super(SOBE_Block, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up_f = nn.Sequential(nn.Conv2d(self.channel1, self.channel2, 7, 1, 3),
                                nn.BatchNorm2d(self.channel2), nn.ReLU(), nn.UpsamplingBilinear2d(scale_factor=2))
        self.up_p = nn.UpsamplingBilinear2d(scale_factor=2)

        self.pcbr1 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.pcbr2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.alpha = nn.Parameter(torch.tensor(1.))

        self.ecbr1 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.ecbr2 = nn.Sequential(nn.Conv2d(self.channel2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.beta = nn.Parameter(torch.tensor(1.))

        self.fusion = nn.Sequential(nn.Conv2d(self.channel2 * 2, self.channel2, 3, 1, 1), nn.BatchNorm2d(self.channel2), nn.ReLU())
        self.memory = ConvGRU(self.channel2)
        self.output_map = nn.Conv2d(self.channel2, 1, 7, 1, 3)
    def forward(self,f_l, f_h, pre,  r: Optional[Tensor]):
        '''

        Args:
            f_l: low level feature
            f_h: high level feature
            pre: low level prediction

        Returns:

        '''
        B, C, H, W = f_l.shape
        residual = f_l

        f_h_up = self.up_f(f_h)
        f_h_up = f_h_up[:,:, :H,:W]

        f_position = f_h_up * f_l   # 为什么可以通过f_h_up * f_l来定位
        f_position = self.pcbr1(f_position)
        f_position = self.alpha * f_position + residual
        f_position = self.pcbr2(f_position)


        pre_up = self.up_p(pre)
        pre_up = pre_up[:,:, :H,:W]

        f_edge = pre_up * f_l
        f_edge = self.ecbr1(f_edge)
        f_edge = self.beta * f_edge + residual
        f_edge = self.ecbr2(f_edge)

        f_refine = torch.cat((f_edge, f_position), dim=1)
        f_refine = self.fusion(f_refine)

        # TODO(cyt): add memory module such as GRU
        # f_refine.size() = (15, 40, 32, 32) ~ (T, C, H, W)
        f_refine, r = self.memory(f_refine, r)

        out_map = self.output_map(f_refine)
        return f_refine, out_map, r


class OutputBlock(nn.Module):
    def __init__(self, in_channels, src_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + src_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
        self.fgr_map = nn.Conv2d(out_channels, 3, 7, 1, 3)
        self.pha_map = nn.Conv2d(out_channels, 1, 7, 1, 3)

    def forward_single_frame(self, x, s):
        x = self.upsample(x)
        x = x[:, :, :s.size(2), :s.size(3)]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)

        fgr = self.fgr_map(x)
        pha = self.pha_map(x)
        return x,pha,fgr

    def forward_time_series(self, x, s):
        B, T, _, H, W = s.shape
        x = x.flatten(0, 1)
        s = s.flatten(0, 1)
        x = self.upsample(x)
        x = x[:, :, :H, :W]
        x = torch.cat([x, s], dim=1)
        x = self.conv(x)
        fgr = self.fgr_map(x)
        pha = self.pha_map(x)
        x = x.unflatten(0, (B, T))
        pha = pha.unflatten(0, (B, T))
        fgr = fgr.unflatten(0, (B, T))
        return x,pha,fgr

    def forward(self, x, s):
        if x.ndim == 5:
            return self.forward_time_series(x, s)
        else:
            return self.forward_single_frame(x, s)


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward_single_frame(self, x):
        return self.conv(x)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        return self.conv(x.flatten(0, 1)).unflatten(0, (B, T))

    def forward(self, x):
        if x.ndim == 5:
            return self.forward_time_series(x)
        else:
            return self.forward_single_frame(x)
