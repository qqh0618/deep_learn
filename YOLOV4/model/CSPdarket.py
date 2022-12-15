import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# 自底向上构建

# ------------------------------------------
# MISH激活函数
# -------------------------------------------
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# -------------------------------------------
# 标准卷积块 ->卷积 + 标准化 + 激活函数
# Conv2d + BatchNormalization + Mish
# BatchNormalization是何凯明大神在resnet提出的，它将每层输出的数据归一化道相同的分布，可以加速收敛，why？
# -------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x

# ------------------------------
# CSPdarknet内部
class Resblock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(Resblock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)

# --------------------------------------
# CSPdarknet结构块完整构建
# 首先利用零填充和步长2*2的卷积核进行卷积压缩
# 建立一个大的残差边，绕过其它残差结构
# 主干部分对num——block进行循环，循环内部是残差结构
# 对于整个CSPdarknet结构，有两条路，一个很大的残差边，和另一个有很多小残差边的结构块
class Resblocak_body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        """
        :param in_channels:输入通道
        :param out_channels: 输出通道
        :param num_blocks: 残差块循环次数
        :param first:
        """
        super(Resblocak_body, self).__init__()
        # ----------------------------------
        # 利用一个步长为2x2的卷积块进行高和宽的压缩
        # ----------------------------------
        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            # --------------------------------------
            # 建立一个大的残差边、绕过另一个边的各种小残差结构
            # --------------------------------------
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.block_conv = nn.Sequential(
                Resblock(channels=out_channels, hidden_channels=out_channels//2),
                BasicConv(out_channels, out_channels, 1)
            )

            self.concat_conv = BasicConv(out_channels*2, out_channels, 1)

        else:
            # --------------------------------------------------
            # 建立一个大的残差边self.split_conv0、这个大残差边绕过了很多的残差结构
            # --------------------------------------------------
            self.split_conv0 = BasicConv(out_channels, out_channels//2, 1)

            # --------------------------------------------------
            # 主干部分会对num_blocks进行循环，循环内部是残差结构
            # --------------------------------------------------
            self.split_conv1 = BasicConv(out_channels, out_channels//2, 1)
            self.block_conv = nn.Sequential(
                *[Resblock(out_channels//2) for _ in range(num_blocks)],
                BasicConv(out_channels//2, out_channels//2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)
        x0 = self.split_conv0(x)
        x1 = self.split_conv1(x)
        x1 = self.block_conv(x1)

        # --------------------------
        # 将大残差边再堆叠回来
        # -------------------------
        x = torch.cat([x1, x0], dim=1)

        # -------------------------
        # 最后对通道数进行整合
        # -------------------------
        x = self.concat_conv(x)

        return x



