"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

#剪枝后保存的模型权重，包含gatelayer层信息
PRUNED_WEIGHT_PATH = '/workspace/mnt/group/algorithm/dingrui/git_file/puringclassfication-train/weights/pruned_network_res18.weights'


__all__ = ['pruned_resnet18']


#读取pruned后，并对权重进行剪枝之后的模型参数，生成对应层的通道结构
def get_pruned_weights(layer_name,layer_number,conv_number,in_or_out,downsample=False):
    
    
    checkpoint = torch.load(PRUNED_WEIGHT_PATH)
    state_dict = checkpoint['state_dict']
    input_channels = []
    output_channels = []
    downsample_input = 0
    downsample_output = 0
    for k, v in state_dict.items():
        # print(k,v.shape)
        if k == 'layer{}.{}.downsample.0.weight'.format(layer_name,layer_number):
            downsample_input = v.shape[1]
            downsample_output = v.shape[0]
            # print(k,v.shape[0],v.shape[1])
        for i in range(1,int(conv_number)+1):
            if k == 'layer{}.{}.conv{}.weight'.format(layer_name,layer_number,i):
                input_channels.append(v.shape[1])
                output_channels.append(v.shape[0])
            elif k == 'layer{}.{}.conv{}.weight'.format(layer_name,layer_number,i):
                input_channels.append(v.shape[1])
                output_channels.append(v.shape[0])
            elif k == 'layer{}.{}.conv{}.weight'.format(layer_name,layer_number,i):
                input_channels.append(v.shape[1])
                output_channels.append(v.shape[0])
    if in_or_out == 1 and downsample == False:
        return input_channels
    elif in_or_out == 2 and downsample == False:
        return output_channels
    elif downsample == 'input':
        return downsample_input
    elif downsample == 'output':
        return downsample_output
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, gate=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes[0], planes[0], stride)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu = nn.ReLU(inplace=True)
        #self.gate1 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.conv2 = conv3x3(inplanes[1], planes[1])
        self.bn2 = nn.BatchNorm2d(planes[1])
        #self.gate2 = GateLayer(planes,planes,[1, -1, 1, 1])
        self.downsample = downsample
        self.stride = stride
        #self.gate = gate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.gate1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.gate2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        # if self.gate is not None:
        #     out = self.gate(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, skip_gate = True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        gate = skip_gate
        self.gate = gate

        self.layer1 = self._make_layer('1',block, 64, layers[0])
        self.layer2 = self._make_layer('2',block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer('3',block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer('4',block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,layer_number,block, planes, blocks, stride=1):

        layers = []
        # layers.append(block(self.inplanes, planes, stride, downsample, gate = gate))

        self.inplanes = planes * block.expansion
        if layer_number == '1':
            # downsample = nn.Sequential(
            #     nn.Conv2d(get_pruned_weights(layer_number,0,3,1,downsample='input'), get_pruned_weights(layer_number,0,3,1,downsample='output'), kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(get_pruned_weights(layer_number,0,3,1,downsample='output')),
            # )
            for i in range(0, blocks):
                if i == 0:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
                else:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
                # elif i == 2:
                #     layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
        elif layer_number == '2':
            downsample = nn.Sequential(
                nn.Conv2d(get_pruned_weights(layer_number,0,3,1,downsample='input'), get_pruned_weights(layer_number,0,3,1,downsample='output'), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(get_pruned_weights(layer_number,0,3,1,downsample='output')),
            )
            for i in range(0, blocks):
                if i == 0:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2),downsample = downsample,stride = stride))
                else:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
        elif layer_number == '3':
            downsample = nn.Sequential(
                nn.Conv2d(get_pruned_weights(layer_number,0,3,1,downsample='input'), get_pruned_weights(layer_number,0,3,1,downsample='output'), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(get_pruned_weights(layer_number,0,3,1,downsample='output')),
            )
            for i in range(0, blocks):
                if i == 0:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2), downsample = downsample,stride = stride))
                else:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
        elif layer_number == '4':
            downsample = nn.Sequential(
                nn.Conv2d(get_pruned_weights(layer_number,0,3,1,downsample='input'), get_pruned_weights(layer_number,0,3,1,downsample='output'), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(get_pruned_weights(layer_number,0,3,1,downsample='output')),
            )
            for i in range(0, blocks):
                if i == 0:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2),downsample = downsample,stride = stride))
                else:
                    layers.append(block(get_pruned_weights(layer_number,i,3,1), get_pruned_weights(layer_number,i,3,2)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # if self.gate:
        #     x=self.gate_skip1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def pruned_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

if __name__ == '__main__':
    model = pruned_resnet18()
    print(model)

    # get_pruned_weights(1,0,3)