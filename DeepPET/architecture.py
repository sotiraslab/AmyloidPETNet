import sys
import os
import shutil
from os import X_OK
from collections import OrderedDict
import time
import numpy as np
from numpy.lib.twodim_base import triu_indices
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, f1_score, roc_curve
import monai
from monai.data import DataLoader
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary


class DeepPETModel(torch.nn.Module):
    def __init__(self):
        super(DeepPETModel, self).__init__()
        pass

    def inheritance_test(self):
        print("DeepPETModel inheritance test passed!")

    def input_gradient_hook(self, gradients):
        """
        cature gradient with respect to input
        """

        # print("triggered input gradient hook")
        self.input_gradient = gradients

    def activation_gradient_hook(self, gradients):
        """
        capture gradient with respect to activation maps
        """

        # print("triggered activation gradient hook")
        self.activation_gradients = gradients

    def get_activation_maps(self, x):
        """
        return the activation maps of the last convolutional block
        """

        return self.activation_maps

    def get_input_gradient(self):
        """
        retrieve gradient with respect to input
        """

        return self.input_gradient

    def get_activation_gradients(self):
        """
        retrieve gradient with respect to activation maps
        """

        return self.activation_gradients


class PreActivationResBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.25)

    def preresidual_gradient_hook(self, gradients):
        """
        cature gradient with respect to input
        """
        self.preresidual_gradients = gradients

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if (not self.training) and (x.requires_grad):
            # print(f"PreActivationResBlock: triggered gradient hook")
            h = x.register_hook(self.preresidual_gradient_hook)
        self.preresidual_activation_maps = out.detach().clone()
        out += x

        return out


class DeepPETEncoder(DeepPETModel):
    def __init__(self):
        super().__init__()

        self.layer0 = self._make_layer(1, 8, stride=1)
        self.layer1 = self._make_layer(8, 16, stride=2)
        self.layer2 = self._make_layer(16, 32, stride=2)
        self.layer3 = self._make_layer(32, 64, stride=2)
        self.layer4 = self._make_layer(64, 128, stride=2)

        self.output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.50),
            torch.nn.Linear(128, 1),
        )

        self.layers = [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

        self.input_gradient = None
        self.gradients = None

    def _make_layer(self, in_planes, out_planes, stride=1):

        layers = [
            nn.Conv3d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        ]
        layers.append(PreActivationResBlock(planes=out_planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        if (not self.training) and (x.requires_grad):
            h = x.register_hook(self.input_gradient_hook)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if (not self.training) and (x.requires_grad):
            # print(f"DeepPETEncoder: triggered gradient hook")
            h = x.register_hook(self.activation_gradient_hook)

        # global average pooling 3d
        x = x.mean(dim=(-3, -2, -1))
        x = self.output(x)

        return x


class PreActivationResBlockGradCAM(nn.Module):
    """
    GradCAM-compatible PreActivaitonResBlock
    """

    def __init__(self, planes):
        super().__init__()
        self.conv1 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=0.25)

    def preresidual_gradient_hook(self, gradients):
        """
        cature gradient with respect to input
        """
        self.preresidual_gradients = gradients

    def forward(self, x):

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if (not self.training) and (x.requires_grad):
            # print(f"PreActivationResBlock: triggered gradient hook")
            h = x.register_hook(self.preresidual_gradient_hook)
        self.preresidual_activation_maps = out.detach().clone()
        out += x

        return out


class DeepPETEncoderGradCAM(DeepPETModel):
    """
    GradCAM-compatible DeepPETEncoder
    """

    def __init__(self):
        super().__init__()

        self.conv_layer0 = self._make_conv_layer(1, 8, stride=1)
        self.conv_layer1 = self._make_conv_layer(8, 16, stride=2)
        self.conv_layer2 = self._make_conv_layer(16, 32, stride=2)
        self.conv_layer3 = self._make_conv_layer(32, 64, stride=2)
        self.conv_layer4 = self._make_conv_layer(64, 128, stride=2)

        self.preres_block0 = self._make_preres_block(8)
        self.preres_block1 = self._make_preres_block(16)
        self.preres_block2 = self._make_preres_block(32)
        self.preres_block3 = self._make_preres_block(64)
        self.preres_block4 = self._make_preres_block(128)

        self.output = torch.nn.Sequential(
            torch.nn.Dropout(p=0.50),
            torch.nn.Linear(128, 1),
        )

        self.input_gradient = None
        self.gradients = None

    def _make_conv_layer(self, in_planes, out_planes, stride=1):

        return nn.Conv3d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def _make_preres_block(self, planes):

        return PreActivationResBlock(planes=planes)

    def forward(self, x0):

        if (not self.training) and (x0.requires_grad):
            h = x0.register_hook(self.input_gradient_hook)

        x0 = self.conv_layer0(x0)
        x1 = self.preres_block0(x0)
        x1 = x1.add(x0)

        x1 = self.conv_layer1(x1)
        x2 = self.preres_block1(x1)
        x2 = x2.add(x1)

        x2 = self.conv_layer2(x2)
        x3 = self.preres_block2(x2)
        x3 = x3.add(x2)

        x3 = self.conv_layer3(x3)
        x4 = self.preres_block3(x3)
        x4 = x4.add(x3)

        x4 = self.conv_layer4(x4)
        x5 = self.preres_block4(x4)
        if (not self.training) and (x5.requires_grad):
            # print(f"DeepPETEncoder: triggered gradient hook")
            h = x5.register_hook(self.activation_gradient_hook)
        # make a copy of tensor and store as activation maps
        self.activation_maps = x5.detach().clone()
        x5 = x5.add(x4)

        # global average pooling 3d
        x5 = x5.mean(dim=(-3, -2, -1))
        x5 = self.output(x5)

        return x5


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.add_module("norm1", nn.BatchNorm3d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv3d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module(
            "conv2",
            nn.Conv3d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate
            )
            self.add_module(f"denselayer{(i + 1)}", layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm3d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool3d(kernel_size=2, stride=2))


class DeepPETDenseNetClassifier(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
    """

    def __init__(
        self,
        no_max_pool=False,
        growth_rate=16,
        block_config=(3, 3, 3, 3),
        bn_size=4,
        drop_rate=0.25,
    ):

        super().__init__()

        # First convolution
        self.features = [
            (
                "conv1",
                nn.Conv3d(
                    1,
                    8,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    bias=False,
                ),
            ),
            ("norm1", nn.BatchNorm3d(8)),
            ("relu1", nn.ReLU(inplace=True)),
        ]
        if not no_max_pool:
            self.features.append(
                ("pool1", nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
            )
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = 8
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{(i + 1)}", block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module(f"transition{(i + 1)}", trans)
                num_features = num_features // 2

        # final batch norm
        self.features.add_module("norm5", nn.BatchNorm3d(num_features))
        # final dense layer
        self.classifier = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool3d(features, output_size=(1, 1, 1)).view(
            features.size(0), -1
        )
        out = self.classifier(out)
        return out
