import torch as torch
import torch.nn as nn
import torchvision
import torch.utils.model_zoo as model_zoo

__all__ = ['VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
'my_vgg':[32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 256, 'M'],
'xx_vgg':[],
'my_mobilenet':[],
}




class VGG(nn.Module):

    def __init__(self, features, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.conv_sub = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=1)
        # self.softmax = nn.Softmax2d()
        # self.linear = nn.Linear(256,10)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)    # (Batch_size, 256, 1, 1)
        x = self.conv_sub(x)    # (Batch_size, 2, 1, 1)
        x = torch.squeeze(x)    # (Batch_size, 2)

        return x

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


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def my_vgg(batch_norm=True, **kwargs):\

    return VGG(make_layers(cfg['my_vgg'], batch_norm=batch_norm), **kwargs)



""""""""""""""""""""""""""""""""""""
"""============ZTK_vgg==========="""
""""""""""""""""""""""""""""""""""""

class Block1(nn.Module):
    # kernel_size=3*3   pad=1
    def __init__(self, in_channel, out_channel, stride=1, padding=1):
        super(Block1,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                              kernel_size=3,stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(self.conv,self.bn,self.relu)

    def forward(self, x):
        x = self.block(x)
        return x


class Block2(nn.Module):
    # kernel_size=3*3   pad=0
    def __init__(self, in_channel, out_channel):
        super(Block2,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=3, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(self.conv, self.bn, self.relu)

    def forward(self, x):
        x = self.block(x)
        return x


class Ztk_vgg(nn.Module):
    def __init__(self, in_channel=3, num_class=2):
        super(Ztk_vgg,self).__init__()
        self.unit1 = Block1(in_channel, 32)
        self.unit2 = Block1(32, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample1 = nn.Sequential(self.unit1,self.unit2,self.maxpool1)

        self.unit3 = Block1(32,64)
        self.unit4 = Block1(64,64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample2 = nn.Sequential(self.unit3,self.unit4,self.maxpool2)

        self.unit5 = Block2(64,128)
        self.unit6 = Block2(128,256)
        self.unit7 = Block2(256,512)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_sample3 = nn.Sequential(self.unit5,self.unit6,self.unit7,self.maxpool3)

        self.conv_1x1 = nn.Conv2d(512,2,kernel_size=1)

        self._initialize_weights()

        # self.feature = nn.Sequential(self.down_sample1,self.down_sample2,self.down_sample3)

    def forward(self, x):
        x = self.down_sample1(x)
        x = self.down_sample2(x)
        x = self.down_sample3(x)
        x = self.conv_1x1(x)
        x = torch.squeeze(x)
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


""""""""""""""""""""""""""""""""""""
"""============MobileNet========="""
""""""""""""""""""""""""""""""""""""

class Depthwise_separable_conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1 ,kernel_size=3, padding=1):
        super(Depthwise_separable_conv,self).__init__()

        self.depthwise = nn.Conv2d(in_channels=in_channel, out_channels=in_channel,
                                   kernel_size=kernel_size, padding=padding,stride=stride,
                                   groups=in_channel)
        self.pointwise = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        self.bn_in = nn.BatchNorm2d(in_channel)
        self.bn_out = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_in(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn_out(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1000):
        super(MobileNet, self).__init__()

        self.conv1 = Block1(in_channel,32,stride=2,padding=1)                           # /2
        self.depth_point_conv1 = Depthwise_separable_conv(32,64,stride=1,padding=1)     # unchanged
        self.depth_point_conv2 = Depthwise_separable_conv(64,128,stride=2,padding=1)    # /2
        # self.depth_point_conv2 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.depth_point_conv3 = Depthwise_separable_conv(128,128,stride=1,padding=1)   # unchanged
        self.depth_point_conv4 = Depthwise_separable_conv(128,256,stride=1,padding=0)   # -2
        self.depth_point_conv5 = Depthwise_separable_conv(256,256,stride=1,padding=1)   # unchanged
        self.depth_point_conv6 = Depthwise_separable_conv(256,512,stride=1,padding=0)   # -2
        self.depth_point_conv7_block = [Depthwise_separable_conv(512,512,stride=1,padding=1)]*5

        self.depth_point_conv8 = Depthwise_separable_conv(512,1024,stride=1,padding=0)  # -2
        self.depth_point_conv9 = Depthwise_separable_conv(1024,1024,stride=2,padding=1) # /2
        # self.depth_point_conv9 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(1024,2,kernel_size=1)

        self.feature = nn.Sequential(self.conv1,
                                     self.depth_point_conv1,
                                     self.depth_point_conv2,
                                     self.depth_point_conv3,
                                     self.depth_point_conv4,
                                     self.depth_point_conv5,
                                     self.depth_point_conv6,
                                     *self.depth_point_conv7_block,
                                     self.depth_point_conv8,
                                     self.depth_point_conv9,
                                     self.conv1x1
                                     )
        self._initialize_weights()
        # self.average_pool = nn.AvgPool2d(kernel_size=7)  # 1*1*1024

        # self.fc = nn.Linear(1024,out_channel)
        # self.softmax = nn.Softmax()
        # self.classifier = nn.Sequential(self.fc,self.softmax)

    def forward(self, x):
        x = self.feature(x)
        # x = x.view(-1, x.shape[1])       # 1024
        # x = self.classifier(x)      # 1000
        # x = self.fc(x)
        # x = self.softmax(x)
        x = torch.squeeze(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)





