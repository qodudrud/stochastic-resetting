import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn.functional as F
from torch.nn import Mish, Embedding, Linear, ModuleList, Sequential
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, PNAConv, global_add_pool, global_mean_pool, GraphNorm, BatchNorm

# models
import torchvision
from torchvision.models import vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn, mobilenet_v2
from vit_pytorch import ViT
import timm

from misc.utils import act_fn

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinResBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinResBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x):
        residual = x
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out += residual
        return out


class ResFCNN(nn.Module):
    def __init__(self):
        super(ResFCNN, self).__init__()
        self.fc_input = nn.Linear(1, 128)
        self.res_block1 = LinResBlock(128, 128)
        self.res_block2 = LinResBlock(128, 128)
        self.res_block3 = LinResBlock(128, 128)
        self.res_block4 = LinResBlock(128, 128)
        self.fc_output = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc_input(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.fc_output(x)
        return x

    
class VCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Dropout(0.2),
            nn.Linear(256*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, args.num_classes))
        
    def forward(self, xb):
        return self.network(xb)

    
class FCNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*32*3, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 250),
            nn.BatchNorm1d(250),
            nn.ReLU(),
            nn.Linear(250, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, args.num_classes),
            nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.network(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def load_model(args):
    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2], args.num_classes)

    def ResNet34():
        return ResNet(BasicBlock, [3, 4, 6, 3], args.num_classes)

    def ResNet50():
        return ResNet(Bottleneck, [3, 4, 6, 3], args.num_classes)

    def ResNet101():
        return ResNet(Bottleneck, [3, 4, 23, 3], args.num_classes)

    def ResNet152():
        return ResNet(Bottleneck, [3, 8, 36, 3], args.num_classes)
    
    def ViT_B16():
        model = ViT(
                    image_size = 32,
                    patch_size = 16,
                    num_classes = args.num_classes,
                    dim = 768,
                    depth = 12,
                    heads = 12,
                    mlp_dim = 4*768,
                    # dropout = 0.1,
                    # emb_dropout = 0.1
                    )
        return model
    
    def ViT_B4():
        model = ViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = args.num_classes,
                    dim = 768,
                    depth = 12,
                    heads = 12,
                    mlp_dim = 4*768,
                    # dropout = 0.1,
                    # emb_dropout = 0.1
                    )
        return model
    
    def ViT_S4():
        model = ViT(
                    image_size = 32,
                    patch_size = 4,
                    num_classes = args.num_classes,
                    dim = 384,
                    depth = 6,
                    heads = 6,
                    mlp_dim = 4*384,
                    # dropout = 0.1,
                    # emb_dropout = 0.1
                    )
        return model
    
    def ViT_T_P16_224(pretrained):
        model = timm.create_model('vit_tiny_patch16_224', pretrained)
        model.head = nn.Linear(model.head.in_features, args.num_classes)
        return model
    
    curr_net, reset_net, best_net = None, None, None

    if args.model == 'vcnn':
        curr_net = VCNN(args).to(args.device)
        reset_net = VCNN(args).to(args.device)
        best_net = VCNN(args).to(args.device)

    elif args.model == 'fcnn':
        curr_net = FCNN(args).to(args.device)
        reset_net = FCNN(args).to(args.device)
        best_net = FCNN(args).to(args.device)

    elif args.model == 'resnet50':
        curr_net = ResNet50().to(args.device)
        reset_net = ResNet50().to(args.device)
        best_net = ResNet50().to(args.device)
    
    elif args.model == 'resnet18':
        curr_net = ResNet18().to(args.device)
        reset_net = ResNet18().to(args.device)
        best_net = ResNet18().to(args.device)
    
    elif args.model == 'resnet34':
        curr_net = ResNet34().to(args.device)
        reset_net = ResNet34().to(args.device)
        best_net = ResNet34().to(args.device)

    elif args.model == 'vit-t-p16-pret':
        curr_net = ViT_T_P16_224(pretrained=True).to(args.device)
        reset_net = ViT_T_P16_224(pretrained=True).to(args.device)
        best_net = ViT_T_P16_224(pretrained=True).to(args.device)

    elif args.model == 'vit-t-p16-rand':
        curr_net = ViT_T_P16_224(pretrained=False).to(args.device)
        reset_net = ViT_T_P16_224(pretrained=False).to(args.device)
        best_net = ViT_T_P16_224(pretrained=False).to(args.device)

    return curr_net, reset_net, best_net


def load_fcnn(args):
    curr_net, reset_net, best_net = None, None, None
    
    if args.model == 'FCNN':
        curr_net = FCNN().to(args.device)
        reset_net = FCNN().to(args.device)
        best_net = FCNN().to(args.device)
    
    return curr_net, reset_net, best_net