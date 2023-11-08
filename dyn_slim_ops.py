import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
__all__ = ['DSConv2d', 'DSTransposeConv2d', 'DSdwConv2d', 'DSpwConv2d', 'DSLinear', 'DSAvgPool2d',
           'DSAdaptiveAvgPool2d']


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DSConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 cat_factor=1):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(DSConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.active_out_channel = out_channels_list[-1]
        self.cat_factor = cat_factor

    def forward(self, x):
        self.running_inc = x.size(1)
        self.running_outc = self.active_out_channel
        if self.cat_factor == 1:
            weight = self.weight[:self.running_outc, :self.running_inc]
        else:
            self.running_inc = x.size(1) // self.cat_factor
            self.weight_chunk = self.weight.chunk(self.cat_factor, dim=1)
            weight = [i[:self.running_outc, :self.running_inc] for i in self.weight_chunk]
            weight = torch.cat(weight, dim=1)

        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc

        return F.conv2d(x,
                        weight,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.running_groups)


class DSTransposeConv2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels_list,  # 输入通道数量的动态变化列表
                 out_channels_list,  # 输出通道数量的动态变化列表
                 kernel_size,
                 stride=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 output_padding=1):
        if not isinstance(in_channels_list, (list, tuple)):
            in_channels_list = [in_channels_list]
        if not isinstance(out_channels_list, (list, tuple)):
            out_channels_list = [out_channels_list]  # 转成list or tuple 的数据格式
        super(DSTransposeConv2d, self).__init__(
            in_channels=in_channels_list[-1],
            out_channels=out_channels_list[-1],
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            output_padding=output_padding)  # 对继承的nn.conv2d 使用输入的参数进行初始化~
        assert self.groups in (1, self.out_channels), \
            'only support regular conv, pwconv and dwconv'
        padding = ((self.stride[0] - 1) + self.dilation[0] * (
                self.kernel_size[0] - 1)) // 2  # 计算same padding 应该填充的宽度
        self.padding = (padding, padding)
        self.output_padding = output_padding
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.active_out_channel = out_channels_list[-1]

    def forward(self, x, *args, **kwargs):
        self.running_inc = x.size(1)
        self.running_outc = self.active_out_channel
        weight = self.weight[:self.running_inc, :self.running_outc]  # 卷积和反卷积的卷积核其对应的输入通道是相反的。
        bias = self.bias[:self.running_outc] if self.bias is not None else None
        self.running_groups = 1 if self.groups == 1 else self.running_outc
        return F.conv_transpose2d(x,
                                  weight,
                                  bias,
                                  self.stride,
                                  self.padding,
                                  self.output_padding,
                                  self.groups)


class DSpwConv2d(DSConv2d):
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 bias=True):
        super(DSpwConv2d, self).__init__(
            in_channels_list=in_channels_list,
            out_channels_list=out_channels_list,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode='zeros')


class DSdwConv2d(DSConv2d):
    def __init__(self,
                 channels_list,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 bias=True):
        super(DSdwConv2d, self).__init__(
            in_channels_list=channels_list,
            out_channels_list=channels_list,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=channels_list[-1],
            bias=bias,
            padding_mode='zeros')


class DSLinear(nn.Linear):
    def __init__(self,
                 in_features_list,
                 out_features,
                 bias=True):
        super(DSLinear, self).__init__(
            in_features=in_features_list[-1],
            out_features=out_features,
            bias=bias)
        self.in_channels_list = in_features_list
        self.out_channels_list = [out_features]
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
            self.running_outc = self.out_features
            return F.linear(x, self.weight, self.bias)
        else:
            self.running_inc = x.size(1)
            self.running_outc = self.out_features
            weight = self.weight[:, :self.running_inc]
            return F.linear(x, weight, self.bias)


class DSAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size, channel_list):
        super(DSAdaptiveAvgPool2d, self).__init__(output_size=output_size)
        self.in_channels_list = channel_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if isinstance(self.channel_choice, tuple):
                self.channel_choice = self.channel_choice[0]  # channel choice 存在问题。
                self.running_inc = torch.matmul(self.channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.channel_choice]
            # self.running_inc = x.size(1)
        else:
            self.running_inc = x.size(1)
        return super(DSAdaptiveAvgPool2d, self).forward(input=x)


class DSAvgPool2d(nn.AvgPool2d):
    def __init__(self, kernel_size, stride, channel_list, padding=0, ceil_mode=True, count_include_pad=False):
        super(DSAvgPool2d, self).__init__(kernel_size=kernel_size, stride=stride, padding=padding,
                                          ceil_mode=ceil_mode, count_include_pad=count_include_pad)
        self.in_channels_list = channel_list
        self.channel_choice = -1
        self.mode = 'largest'
        self.prev_channel_choice = None
        self.in_channels_list_tensor = torch.from_numpy(
            np.array(self.in_channels_list)).float().cuda()

    def forward(self, x):
        if self.mode == 'dynamic':
            if self.prev_channel_choice is None:
                self.prev_channel_choice = self.channel_choice
            if isinstance(self.prev_channel_choice, tuple):
                self.prev_channel_choice = self.prev_channel_choice[0]
                self.running_inc = torch.matmul(self.prev_channel_choice, self.in_channels_list_tensor)
            else:
                self.running_inc = self.in_channels_list[self.prev_channel_choice]
            self.prev_channel_choice = None
        else:
            self.running_inc = x.size(1)
        return super(DSAvgPool2d, self).forward(input=x)