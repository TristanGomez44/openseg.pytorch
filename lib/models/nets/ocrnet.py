##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pdb
import torch
import torch.nn as nn
from torch.nn import functional as F

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper

import argparse
import sys
class SpatialOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self,configer):
        self.inplanes = 128
        super(SpatialOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )

        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
                                                  key_channels=256,
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.conv_3x3(x[-1])
        context = self.spatial_context_head(x, x_dsn)
        x_feat = self.spatial_ocr_head(x, context)
        x = self.head(x_feat)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        class_feat = representativeVectors(x_feat,x)
        x_feat_resh = x_feat.view(x_feat.size(0),x_feat.size(1),-1).permute(0,2,1)

        scal_prod = (class_feat.unsqueeze(1) * x_feat_resh.unsqueeze(2)).sum(dim=-1)
        class_feat_norm = torch.sqrt(torch.pow(class_feat,2).sum(dim=-1)).unsqueeze(1)
        x_feat_resh_norm = torch.sqrt(torch.pow(x_feat_resh,2).sum(dim=-1)).unsqueeze(2)
        cos_sim = scal_prod/(class_feat_norm*x_feat_resh_norm)
        weights = torch.softmax(cos_sim,dim=-1)
        x_new_feat = (weights.unsqueeze(3)*class_feat.unsqueeze(1)).sum(dim=2)
        x_new_feat = x_new_feat.permute(0,2,1).view(x_feat.size(0),x_feat.size(1),x_feat.size(2),x_feat.size(3))
        x = self.head(x_new_feat)

        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x_dsn, x

def representativeVectors(x_feat,preds):

    nbVec = preds.size(1)

    x_feat = x_feat.permute(0,2,3,1).reshape(x_feat.size(0),x_feat.size(2)*x_feat.size(3),x_feat.size(1))
    preds = preds.permute(0,2,3,1).reshape(preds.size(0),preds.size(2)*preds.size(3),preds.size(1))
    norm = torch.sqrt(torch.pow(x_feat,2).sum(dim=-1)) + 0.00001

    repreVecList = []

    for i in range(nbVec):
        _,ind = preds[:,:,i].max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x_feat.size(0)).unsqueeze(1),ind]
        raw_reprVec = x_feat[torch.arange(x_feat.size(0)).unsqueeze(1),ind]
        sim = (x_feat*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)
        simNorm = sim/sim.sum(dim=1,keepdim=True)
        reprVec = (x_feat*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec.unsqueeze(1))

    repreVecList = torch.cat(repreVecList,dim=1)

    return repreVecList

class ASPOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(ASPOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096]
        else:
            in_channels = [1024, 2048]

        # we should increase the dilation rates as the output stride is larger
        from lib.models.modules.spatial_ocr_block import SpatialOCR_ASP_Module
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048,
                                                  hidden_features=256,
                                                  out_features=256,
                                                  num_classes=self.num_classes,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.asp_ocr_head(x[-1], x_dsn)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x_dsn, x
