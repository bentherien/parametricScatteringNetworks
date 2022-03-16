"""Contains all the 'top' pytorch NN.modules for this project

Authors: Benjamin Therien, Shanel Gauthier

Functions: 
    conv3x3              -- 3x3 convolution with padding
    countLearnableParams -- returns the amount of learnable parameters in this model

Classes: 
    sn_CNN         -- CNN fitted for scattering input
    sn_LinearLayer -- Linear layer fitted for scattering input
    sn_MLP         -- Multilayer perceptron fitted for scattering input
    BasicBlock     -- Standard wideresnet basicblock
    Resnet50       --Pretrained resnet-50 on ImageNet
"""

from torchvision import models

import torch.nn as nn


class sn_MLP(nn.Module):
    """
       Multilayer perceptron fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        super(sn_MLP,self).__init__()
        self.num_classes = num_classes

        fc1 =  nn.Linear(int(3*M_coefficient*N_coefficient*n_coefficients), 512)

        self.layers = nn.Sequential(
            nn.BatchNorm2d(self.n_coefficients*3, eps=1e-5, affine=True),
            fc1,
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward pass"""
        x = x.view(x.shape[0], -1)
        return self.layers(x)


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    Standard wideresnet basicblock
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

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

        return out


class sn_CNN(nn.Module):
    """
    CNN fitted for scattering input
    Model from: https://github.com/kymatio/kymatio/blob/master/examples/2d/cifar_small_sample.py 
    """
    def __init__(self, in_channels, k=8, n=4, num_classes=10, standard=False):
        super(sn_CNN, self).__init__()

        self.bn0 = nn.BatchNorm2d(in_channels*3,eps=1e-5,affine=True)

        self.inplanes = 16 * k
        self.ichannels = 16 * k
        self.in_channels = in_channels
        self.num_classes =num_classes
        in_channels = in_channels * 3
        if standard:

            self.init_conv = nn.Sequential(
                nn.Conv2d(3, self.ichannels,
                          kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.layer1 = self._make_layer(BasicBlock, 16 * k, n)
            self.standard = True
        else:
            self.K = in_channels
            self.init_conv = nn.Sequential(
                nn.BatchNorm2d(in_channels, eps=1e-5, affine=False),
                nn.Conv2d(in_channels, self.ichannels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.ichannels),
                nn.ReLU(True)
            )
            self.standard = False

        self.layer2 = self._make_layer(BasicBlock, 32 * k, n)
        self.layer3 = self._make_layer(BasicBlock, 64 * k, n)
        self.avgpool = nn.AdaptiveAvgPool2d(2)
        self.fc = nn.Linear(64 * k * 4, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.standard:
            pass
        x = self.bn0(x)
        x = self.init_conv(x)
        if self.standard:
            x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class sn_LinearLayer(nn.Module):
    """
    Linear layer fitted for scattering input
    """
    def __init__(self, num_classes=10, n_coefficients=81, M_coefficient=8, N_coefficient=8):
        super(sn_LinearLayer, self).__init__()
        self.n_coefficients = n_coefficients
        self.num_classes = num_classes

        self.fc1 = nn.Linear(int(3*M_coefficient*N_coefficient*n_coefficients), num_classes)
        self.bn0 = nn.BatchNorm2d(self.n_coefficients*3, eps=1e-5, affine=True)

    def forward(self, x):
        x = self.bn0(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


class sn_Resnet50(nn.Module):
    """
    Pretrained model on ImageNet
    Architecture: ResNet-50
    """
    def __init__(self, num_classes=10):
        super(sn_Resnet50, self).__init__()
        self.model_ft = models.resnet50(pretrained=True)
        num_ftrs = self.model_ft.fc.in_features
        self.model_ft.fc =  nn.Linear(num_ftrs, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.model_ft(x)
        return x

