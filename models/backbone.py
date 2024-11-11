  
import numpy as np
import torch
from torch import nn 
import torchvision.models as models
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = 'cuda'

class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Additional stage
        self.conv4 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Third Stage
        self.conv1_stage3 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1_stage3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2_stage3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2_stage3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3_stage3 = nn.Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3_stage3 = nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_stage3 = nn.ReLU(inplace=True)
        self.downsample_stage3 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        #     nn.Linear(2048, 4,bias = True)
        # )
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += identity
        x = self.relu(x)
        x1 = x
        # Additional stage
        identity = self.downsample2(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        
        x += identity
        x = self.relu(x)
        x2 = x

        # Third Stage
        identity_stage3 = self.downsample_stage3(x)

        x = self.conv1_stage3(x)
        x = self.bn1_stage3(x)
        x = self.relu_stage3(x)

        x = self.conv2_stage3(x)
        x = self.bn2_stage3(x)
        x = self.relu_stage3(x)

        x = self.conv3_stage3(x)
        x = self.bn3_stage3(x)

        x += identity_stage3
        x = self.relu_stage3(x)
        x3 = x
        return x1,x2,x3
        # y = self.head[0](x).view(x.size(0), -1)
        # y = self.head[1](y)
        # return x1,x2,x3,y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Backbone2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        )
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        # Additional stage
        self.conv4 = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn6 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        identity = self.downsample(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        
        x += identity
        x = self.relu(x)
        x1 = x
        # Additional stage
        identity = self.downsample2(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        
        x += identity
        x = self.relu(x)
        x2 = x
        return x1,x2

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
# Usage example:
# backbone = Backbone().to(device)
# x  =torch.randn([1,3,224,224]).to(device)
# x1,x2,x3,y = backbone(x)
# print(x1.shape,x2.shape,x3.shape)
# print(y)