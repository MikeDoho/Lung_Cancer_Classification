import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, dropout_rate=0.05):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.dropout3d_bottleneck = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.dropout3d_bottleneck(out)  # added dropout
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.dropout3d_bottleneck(out)  # added dropout
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.dropout3d_bottleneck(out)  # added dropout
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.6)

        # the num 2 will need to be change if number of class change!!!!
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)

        self.conv_pred = nn.Sequential(
            nn.Conv3d(
                2048*1,
                512,
                kernel_size=3,
                stride=(2, 2, 2),  # [10,19,10] pad to [12,21,12] then down to [4,7,4]
                padding=(1, 1, 1),
            ),

            # nn.Dropout3d(p=0.05),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                512,
                256,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),

            # nn.Dropout3d(p=0.05),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),

            nn.Conv3d(
                256,
                128,
                kernel_size=3,
                stride=(2, 2, 2),  # [4,7,4] pad to [6,9,6] then down to [2,3,2]
                padding=(1, 1, 1),
            ),

            # nn.Dropout3d(p=0.05),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                128,
                128,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),

            # nn.Dropout3d(p=0.05),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [2,3,2] to [1,1,1]
            nn.Flatten(),  # [batch_size,128,1,1,1] to [batch_size,128]

            nn.Dropout(p=0.6),
            nn.Linear(128, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                dropout_downsample = nn.Dropout3d(p=0.20)
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    dropout_downsample,
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):

        CT = x[:, 0, :].unsqueeze(1)
        # PET = x[:, 1, :].unsqueeze(1)

        x = self.conv1(CT)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # print('ct ', x.view(x.size()))
        # x1 = x
        #
        # x = self.conv1(PET)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # # x = self.avgpool(x)
        # # print('pet ', x.view(x.size()))
        # x2 = x
        #
        # x = torch.cat([x1, x2], dim=1)
        # print(f"size of tensor: {x.size()}")


        # del x1
        # del x2
        del CT
        # del PET


        x = self.conv_pred(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
