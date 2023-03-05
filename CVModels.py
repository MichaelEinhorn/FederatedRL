import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

import resnet
import core

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ValueHead(nn.Module):
    def __init__(self, n_in=64, n_out=1, detach_head=False, layers=1, hidden_scale=2):
        super().__init__()
            
        self.detach_head = detach_head

        if layers == 1:
            self.summary = nn.Linear(n_in, n_out)
        elif layers == 2:
            self.summary = nn.Sequential(nn.Linear(n_in, int(n_in * hidden_scale)), nn.ReLU(), nn.Linear(int(n_in * hidden_scale), n_out))
        else:
            raise NotImplementedError("Only 1 or 2 layers are supported for the value head.")

    def forward(self, x):
        if self.detach_head:
            output = x.detach()
        else:
            output = x

        output = self.summary(output)
        return output

class CNNAgent(nn.Module):
    def __init__(self, obs_shape, num_actions, channels=32, layers=[2,2,2,2], scale=[1,1,1,1], vheadLayers=1):
        super(CNNAgent, self).__init__()

        img_size = 64

        # input layer
        self.conv1 = nn.Conv2d(obs_shape[-1], channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        img_size = img_size // 2

        # defaults if layer count is 0
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()
        self.layer3 = nn.Identity()
        self.layer4 = nn.Identity()
        self.flatten = nn.Flatten(start_dim=1)

        # residual blocks
        if layers[0] != 0:
            out_channels = channels * scale[0]
            self.layer1 = resnet.makeLayerBlock(resnet.BasicBlock, channels, out_channels, layers[0])
            channels = out_channels
        if layers[1] != 0:
            out_channels = channels * scale[1]
            self.layer2 = resnet.makeLayerBlock(resnet.BasicBlock, channels, out_channels, layers[1], stride=2)
            channels = out_channels
            img_size = img_size // 2
        if layers[2] != 0:
            out_channels = channels * scale[2]
            self.layer3 = resnet.makeLayerBlock(resnet.BasicBlock, channels, out_channels, layers[2], stride=2)
            channels = out_channels
            img_size = img_size // 2
        if layers[3] != 0:
            out_channels = channels * scale[3]
            self.layer4 = resnet.makeLayerBlock(resnet.BasicBlock, channels, out_channels, layers[3], stride=2)
            channels = out_channels
            img_size = img_size // 2

        print(img_size, channels)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(channels * resnet.BasicBlock.expansion * img_size * img_size, num_actions)

        self.valueHead = ValueHead(n_in=channels * resnet.BasicBlock.expansion * img_size * img_size, n_out=1, layers=vheadLayers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        x = self.flatten(x)
        l = self.linear(x)
        v = self.valueHead(x)

        return l, v
    
from timm.models.vision_transformer import VisionTransformer
class ViTValue(nn.Module):
    def __init__(self, img_size=64, patch_size=4, num_classes=15, depth=3, num_heads=4, embed_dim=16, mlp_ratio=4, valueHeadLayers=1):
        super().__init__()
        from CVModels import ValueHead
        self.model = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=num_classes, depth=depth, num_heads=num_heads, embed_dim=embed_dim, mlp_ratio=mlp_ratio)
        self.value = ValueHead(n_in=embed_dim, n_out=1, layers=valueHeadLayers)
    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, True) # pre logits doesn't apply head layer yet
        l = self.model.head(x)
        v = self.value(x)
        return l, v