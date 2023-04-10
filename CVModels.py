from copy import deepcopy

import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

import resnet
from procgen_model import ImpalaModel, NatureModel
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
    def __init__(self, obs_shape, num_actions, channels=32, layers=None, scale=None, vheadLayers=1):
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
    
class ViTValue(nn.Module):
    def __init__(self, img_size=64, patch_size=4, num_classes=15, depth=3, num_heads=4, embed_dim=16, mlp_ratio=4, valueHeadLayers=1):
        super().__init__()
        self.model = VisionTransformer(img_size=img_size, patch_size=patch_size, num_classes=num_classes, depth=depth, num_heads=num_heads, embed_dim=embed_dim, mlp_ratio=mlp_ratio)
        self.value = ValueHead(n_in=embed_dim, n_out=1, layers=valueHeadLayers)
    
    def forward(self, x):
        x = self.model.forward_features(x)
        x = self.model.forward_head(x, True) # pre logits doesn't apply head layer yet
        l = self.model.head(x)
        v = self.value(x)
        return l, v
    
class ImpalaValue(nn.Module):
    def __init__(self, channels=3, num_classes=15, valueHeadLayers=1):
        super().__init__()
        self.model = ImpalaModel(in_channels=3)
        self.linear = nn.Linear(self.model.output_dim, num_classes)
        self.value = ValueHead(n_in=self.model.output_dim, n_out=1, layers=valueHeadLayers)

    def forward(self, x):
        x = self.model(x)
        l = self.linear(x)
        v = self.value(x)
        return l, v
    
class NatureValue(nn.Module):
    def __init__(self, channels=3, num_classes=15, valueHeadLayers=1):
        super().__init__()
        self.model = NatureModel(in_channels=channels)
        self.linear = nn.Linear(self.model.output_dim, num_classes)
        self.value = ValueHead(n_in=self.model.output_dim, n_out=1, layers=valueHeadLayers)

    def forward(self, x):
        x = self.model(x)
        l = self.linear(x)
        v = self.value(x)
        return l, v
    
def printParams(modelList):
    print(len(list(modelList[0].parameters())))
    for paramList in list(zip(*[list(submodel.parameters()) for submodel in modelList])):
        for param in paramList:
            print(param.shape, param.device, param.dtype, param.requires_grad)
        
        print("data equality ", torch.all(paramList[0].data == paramList[1].data), "ref equality ", id(paramList[0]) == id(paramList[1]))

@torch.no_grad()
def avgSync(modelList):
    # counts upload and download, not including log(N) communication complexity for a network architecture
    # data is only for a single clients uploads and downloads
    stats = {"sync/comms": 2, "sync/data": 0}
    for paramList in list(zip(*[list(submodel.parameters()) for submodel in modelList])):
        avgParam = torch.mean(torch.stack([param.data for param in paramList], dim=0), dim=0).detach()
        for param in paramList:
            stats["sync/data"] += 2 * param.data.numel() * param.data.element_size() / 1048576 # megabyte
            # in place copy
            param.data.copy_(avgParam)
    return stats

# adds the sums of all the diffs from the last global params
class sumSync:
    def __init__(self):
        self.refParams = []

    @torch.no_grad()
    def __call__(self, modelList):
        stats = {"sync/comms": 2, "sync/data": 0}
        first = len(self.refParams) == 0 # first time no global params to sum diffs from
        for i, paramList in enumerate(list(zip(*[list(submodel.parameters()) for submodel in modelList]))):
            if first: # average first time
                sumParam = torch.mean(torch.stack([param.data for param in paramList], dim=0), dim=0).detach()
                self.refParams.append(sumParam)
            else:
                ref = self.refParams[i]
                sumParam = torch.sum(torch.stack([param.data - ref.data for param in paramList], dim=0), dim=0).detach() # sum of diffs
                sumParam += ref.data # add reference back

            for param in paramList:
                stats["sync/data"] += 2 * param.data.numel() * param.data.element_size() / 1048576 # megabyte
                # in place copy
                param.data.copy_(sumParam)
        return stats
    
class VectorModelValue(nn.Module):
    def __init__(self, model, n=2, syncFunc=avgSync):
        super().__init__()
        self.modelList = nn.ModuleList([deepcopy(model) for _ in range(n)])
        self.n = n
        self._syncFunc = syncFunc

        self.stats = {"sync/comms": 0, "sync/data": 0}
    
    # Model x Batch x Data
    # https://discuss.pytorch.org/t/is-it-possible-to-execute-two-modules-in-parallel-in-pytorch/54866
    # if there is capacity, the GPU calls should execute in parallel asynchronously
    def forward(self, x):
        lList = []
        vList = []
        for i in range(self.n):
            l, v = self.modelList[i](x[i])
            lList.append(l)
            vList.append(v)
        l = torch.stack(lList, dim=0)
        v = torch.stack(vList, dim=0)
        return l, v
    
    def sync(self):
        if self.n == 1:
            return
        stat = self._syncFunc(self.modelList)
        core.update_dict_add(self.stats, stat)
