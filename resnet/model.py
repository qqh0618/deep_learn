import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础残差模块
class BasicBlock(nn.Module):
    expansion = 1 # 同一层输入和输出相差是一倍的关系
    def __init__(self, in_channel, out_channel, stride, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels = out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity  # 残差相加
        x = self.relu(x)

class Bottleneck(nn.Module):
    expansion = 4  # 同一层输入和输出相差是4倍的关系
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channel*(width_per_group/64.))*groups
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,groups=groups,
                               kernel_size=1,stride=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self,x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x += identity
        x = self.relu(x)

class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000,include_top=True,groups=1,width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 目的是构建四个层主要网络，block是残差块，第二个参数是每层网络输入尺寸，第四个参数是当前网络重复构建次数
        self.layer1 = self._make_layer(block, 64, blocks_num[0])


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride!=1 or self.in_channel!=channel*block.expansion:
            if stride!=1 or self.in_channel!=channel*block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channel, channel*block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(channel*block.expansion)
                )
        layers = []

        layers.append(block(self.in_channel,
                     channel,
                     downsample=downsample,
                           stride=stride,
                           groups=self.groups,
                           width_per_group=self.width_per_group))
        self.in_channel = channel*block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x



