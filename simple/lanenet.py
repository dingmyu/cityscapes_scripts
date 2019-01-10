#--coding:utf-8--
import torch.nn as nn
import torch
import math

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, stride=1, k_size=3, padding=1, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                            padding=padding, stride=stride, bias=bias, dilation=dilation)

        if with_bn:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs



class Net(nn.Module):
    def __init__(self):
        # nn.Module子类的函数必须在构造函数中执行父类的构造函数
        # 下式等价于nn.Module.__init__(self)
        super(Net, self).__init__()
        self.conv1 = conv2DBatchNormRelu(3, 64, 2) 
        self.conv2a1 = conv2DBatchNormRelu(64, 64)
        self.conv2a2 = conv2DBatchNormRelu(64,128)
        self.conv2a_strided = conv2DBatchNormRelu(128,128,2)
        
        self.conv3 = conv2DBatchNormRelu(128,64,1)
        self.conv4 = conv2DBatchNormRelu(64,64,1)        
        self.conv6 = conv2DBatchNormRelu(64,64,1)
        self.conv8 = conv2DBatchNormRelu(64,64,1)
        self.conv9 = conv2DBatchNormRelu(64,16,1)
        self.conv11 = conv2DBatchNormRelu(16,2,1)
        
        self.conv4_ = conv2DBatchNormRelu(64,64,1)        
        self.conv6_ = conv2DBatchNormRelu(64,64,1)
        self.conv8_ = conv2DBatchNormRelu(64,64,1)
        self.conv9_ = conv2DBatchNormRelu(64,16,1)
        self.conv11_ = conv2DBatchNormRelu(16,2,1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2a1(x)
        x = self.conv2a2(x)
        x = self.conv2a_strided(x)
        x = self.conv3(x)
        
        x = self.conv4(x)
        x_ = self.conv4_(x)
        x = nn.Upsample(size=(56,56),mode='bilinear')(x)
        x = self.conv6(x)
        x = self.conv8(x)
        x = nn.Upsample(size=(112,112),mode='bilinear')(x)
        x = self.conv9(x)
        x = self.conv11(x)
        
        
        x_ = nn.Upsample(size=(56,56),mode='bilinear')(x_)
        x_ = self.conv6_(x_)
        x_ = self.conv8_(x_)
        x_ = nn.Upsample(size=(112,112),mode='bilinear')(x_)
        x_ = self.conv9_(x_)
        x_ = self.conv11_(x_)        
        return x, x_

#net = Net()
#print(net)
