from __future__ import absolute_import

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from typing import Any, List, Optional, Type, Union
from torch.ao.quantization import fuse_modules

__all__ = ['resnet']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.add_relu = torch.nn.quantized.FloatFunctional()


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.add_relu.add_relu(out, residual)


        return out
    
    def _fuse_model(self, is_qat: Optional[bool] = None) -> None:
        fuse_modules(self, [["conv1", "bn1", "relu"], ["conv2", "bn2"]],inplace=True)
        if self.downsample:
            fuse_modules(self.downsample, ["0", "1"], inplace=True)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        out = self.skip_add_relu(out, residual)


        return out
    
    def _fuse_model(self, is_qat: Optional[bool] = None) -> None:
        fuse_modules(
            self, [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3"]], inplace=True
        )
        if self.downsample:
            fuse_modules(self.downsample, ["0", "1"], inplace=True)


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, planes=None,q = False):
        super(ResNet, self).__init__()
        if planes is None:
            planes = [32, 64, 128]
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=54 else BasicBlock
        # ========== according to the GraSP code, we double the #filter here ============
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes[0], n)
        self.layer2 = self._make_layer(block, planes[1], n, stride=2)
        self.layer3 = self._make_layer(block, planes[2], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)

        self.fc = nn.Linear(planes[2] * block.expansion, num_classes)
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
      
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    

        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x) 
        try:
            kernel_size = int(x.size()[3])
        except:
            kernel_size = 8
        x = F.avg_pool2d(x,kernel_size)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)

        return x
    
    def _fuse_model(self, is_qat: Optional[bool] = None) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        fuse_modules(self, ["conv1", "bn1", "relu"], inplace=True)
        for m in self.modules():
            if type(m) is Bottleneck or type(m) is BasicBlock:
                m._fuse_model(is_qat)




def resnet(**kwargs):
    """
    Constructs a ResNet models.
    """
    return ResNet(**kwargs)