import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .LA_Position import LA_Position
from .POBEVM_Decoder import SOBE
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

class POBEVM(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.lr_aspp = LA_Position(960, 128)
            self.SOBE = SOBE([16, 24, 40, 128])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src  #原图
        B,T,C,H,W = src.shape
        f1, f2, f3, f4 = self.backbone(src_sm)
        f4, p4 = self.lr_aspp(f4)    #改变通道后，对每一个通道乘以一个权重
        # src_sm.size = (1, 15, 3, 256, 256) ~ (B, T, C, H, W)
        fgr_feature, fgr_residual, p0, p1, p2, p3 = self.SOBE(src_sm, f1, f2, f3, f4, p4, r1, r2, r3, r4)
        if downsample_ratio != 1:
            fgr_residual, p0 = self.refiner(src, src_sm, fgr_residual, p0, fgr_feature)

        p4 = F.interpolate(p4.flatten(0, 1), size=(H,W), mode='bilinear', align_corners=True).unflatten(0, (B, T)).clamp(0., 1.)

        p3 = F.interpolate(p3.flatten(0, 1), size=(H,W), mode='bilinear', align_corners=True).unflatten(0, (B, T)).clamp(0., 1.)
        p2 = F.interpolate(p2.flatten(0, 1), size=(H,W), mode='bilinear', align_corners=True).unflatten(0, (B, T)).clamp(0., 1.)
        p1 = F.interpolate(p1.flatten(0, 1), size=(H,W), mode='bilinear', align_corners=True).unflatten(0, (B, T)).clamp(0., 1.)

        fgr = fgr_residual + src
        fgr = fgr.clamp(0., 1.)
        p0 = p0.clamp(0., 1.)

        return fgr, p0, p1, p2, p3, p4

        #hid, *rec = self.SOBE(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)  #hid即output模块的Relu层输出的隐藏变量，而rec表示各个GRU模块的重置门
        
        # if not segmentation_pass:
        #     fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)   #fgr_residual = F-I，I就是原图，所以下面加回来了src
        #     if downsample_ratio != 1:
        #         fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
        #     fgr = fgr_residual + src
        #     fgr = fgr.clamp(0., 1.)
        #     pha = pha.clamp(0., 1.)
        #     return [fgr, pha, *rec]
        # else:
        #     seg = self.project_seg(hid)
        #     return [seg, *rec]

        return 0

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
