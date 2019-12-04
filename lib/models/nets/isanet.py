##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Microsoft Research
## Author: RainbowSecret, LangHuang
## Copyright (c) 2019
## yuyua@microsoft.com
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper


class ISANet(nn.Module):
    """
    Interlaced Sparse Self-Attention for Semantic Segmentation,
    Huang Lang and Yuan Yuhui and Guo Jianyuan and Zhang Chao and Chen Xilin and Wang Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(ISANet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        bn_type = self.configer.get('network', 'bn_type')
        factors = self.configer.get('network', 'factors')
        from lib.models.modules.isa_block import ISA_Module
        self.isa_head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            ISA_Module(in_channels=512, key_channels=256, value_channels=512, 
                out_channels=512, down_factors=factors, dropout=0.05, bn_type=bn_type),
        )
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=bn_type),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn_head(x[-2])
        x = self.isa_head(x[-1])
        x = self.cls_head(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x


class Asp_ISANet(nn.Module):
    """
    Interlaced Sparse Self-Attention for Semantic Segmentation,
    Huang Lang and Yuan Yuhui and Guo Jianyuan and Zhang Chao and Chen Xilin and Wang Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(Asp_ISANet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        from lib.models.modules.isa_block import ASP_ISA_Module
        self.context = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            ASP_ISA_Module(512, 256, bn_type=self.configer.get('network', 'bn_type')),
        )
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x_):
        x = self.backbone(x_)
        aux_x = self.dsn_head(x[-2])
        x = self.context(x[-1])
        x = self.cls_head(x)
        aux_x = F.interpolate(aux_x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return aux_x, x