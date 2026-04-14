# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1: 提取基础线条
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))  # -> 80x30

        # Layer 2: 提取局部形状
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))  # -> 40x15

        # Layer 3: 提取高层语义
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))  # -> 20x7 (注意：7.5取7)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 7, 1024),
            nn.Dropout(0.5),
            nn.ReLU())

        self.rfc = nn.Linear(1024, captcha_setting.MAX_CAPTCHA * captcha_setting.ALL_CHAR_SET_LEN)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out