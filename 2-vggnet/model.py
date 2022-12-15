import torch.nn as nn
import torch.nn.functional as F
import torch
class VGG(nn.Module):
    def __init__(self, name="vgg16", classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H, W = 112
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # H,W = 56
        )
        if name=="vgg16":
            self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 28
            )
        elif name=="vgg19":
            self.conv2 = nn.Sequential(
                nn.Conv2d(128, 256,  kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256,  kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        if name=="vgg16":
            self.conv3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 14
            )
        elif name=="vgg19":
            self.conv3 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        if name=="vgg16":
            self.conv4 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)  # 7
            )
        elif name=="vgg19":
            self.conv4 = nn.Sequential(
                nn.Conv2d(5112, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, classes)
        )
        if init_weights:
            self._initialize_weights()
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # N*7*7*512
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)