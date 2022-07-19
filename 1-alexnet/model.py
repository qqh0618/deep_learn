import torch.nn as nn
import torch
import torch.nn.functional as F
class AlexNet(nn.Module):
    # 可以传入参数，方便构造模型
    def __init__(self, classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 模型构建有两周方式，一种先定义网络层，再用forward串起来
        # 第二种是利用Sequential直接串起来

        # --------------------------------------------
        # 第一种
        # --------------------------------------------
        # Conv2d(in_channels: int, 输入通道
        #         out_channels: int,  输出通道
        #         kernel_size: _size_2_t,  卷积核大小
        #         stride: _size_2_t = 1,  步长
        #         padding: _size_2_t = 0,  填充
        #         dilation: _size_2_t = 1,
        #         groups: int = 1,
        #         bias: bool = True,  偏差
        #         padding_mode: str = 'zeros'  填充方式默认零填充

        # 经卷积后的矩阵尺寸大小计算公式为  N = (W-F+2P)/S+1, W为输入图片大小,F为卷积核大小，S为步长,P为填充大小

        # 224*224*3--> 55*55*48
        self.conv1 = nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2)

        # Relu() inplace默认为False
        self.relu1 = nn.ReLU(inplace=True)  # 因为后面用到的都是相同的所以只定义了一个，可以每个都定义

        # MaxPool2d() 最大池化
        # 55*55*48-->27*27*48
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 基本卷积块就由 Conv2d  Relu MaxPool2d三部分组成

        # 27*27*48 --> 27*27*128
        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # 27*27*128 --> 13*13*128
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 13*13*128 --> 13*13*192
        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        # 13*13*192 --> 13*13*192
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        # 13*13*192 --> 13*13*128
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)

        # 13*13*128 --> 6*6*128
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # 实际tensor的存储格式为(N,C,H,W),N->bath_size,C->channel,此时为(4,128,6,6)
        # Dropout(p=0.5) 随机失活比例
        self.drop1 = nn.Dropout(p=0.5)
        self.line1 = nn.Linear(128*6*6, 2048)
        self.relu6 = nn.ReLU(inplace=True)

        self.drop2 = nn.Dropout(p=0.5)
        self.line2 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU(inplace=True)

        self.line3 = nn.Linear(2048, classes)

        if init_weights:
            self._initialize_weights()  # 调用的父类函数


    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        # N,C,H,W  ---- 4,6,6,128
        x = torch.flatten(x, start_dim=1) # 将1维到最后一维的数据展开成一维，即如果x的维度维(3,4,5,6)-->(3,120)
        # x->(4,6*6*128)
        x = self.drop1(x)
        x = self.line1(x)
        x = self.relu6(x)
        x = self.drop2(x)
        x = self.line2(x)
        x = self.relu7(x)
        x = self.line3(x)

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
