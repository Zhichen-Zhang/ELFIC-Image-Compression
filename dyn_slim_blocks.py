import torch.nn as nn
from dyn_slim_ops import DSConv2d, DSTransposeConv2d
from Base.base_block import BasicBlock, set_exist_attr
from Base.base_utils import get_net_device
import torch.nn.functional as F
import torch

class IBasicConv2D(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1,
                 act_layer=None,
                 bias=True,
                 ):
        super(IBasicConv2D, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation
        self.act_func = act_layer

        # Basic 2D convolution
        self.conv = DSConv2d(in_channels_list,
                             out_channels_list,
                             kernel_size=kernel_size,
                             stride=stride,
                             dilation=(dilation, dilation),
                             bias=bias)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None
        self.initialize()
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

    @property
    def config(self):
        return {
            'name': IBasicConv2D.__name__,
            'in_channel_list': self.in_channels_list,
            'out_channel_list': self.out_channels_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return IBasicConv2D(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        self.active_out_channel = self.out_channels_list[self.channel_choice]
        sub_layer = IBasicConv2D(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation, act_layer=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
        # act parameters to be added ~
        return sub_layer


class IBasicTransposeConv2D(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride=1,
                 act_layer=None,
                 bias=True,
                 output_padding=(1, 1),
                 ):
        super(IBasicTransposeConv2D, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.act_func = act_layer

        self.conv = DSTransposeConv2d(in_channels_list,
                                      out_channels_list,
                                      kernel_size=kernel_size,
                                      stride=(stride, stride),
                                      bias=bias,
                                      output_padding=output_padding)
        if act_layer == "LeakyRelu":
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif act_layer == "Relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

        self.initialize()
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.conv(x)
        if self.act is not None:
            x = self.act(x)
        return x

    @property
    def config(self):
        return {
            'name': IBasicTransposeConv2D.__name__,
            'in_channel_list': self.in_channels_list,
            'out_channel_list': self.out_channels_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': self.act_func,
        }

    @staticmethod
    def build_from_config(config):
        return IBasicTransposeConv2D(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        self.active_out_channel = self.out_channels_list[self.channel_choice]
        sub_layer = IBasicTransposeConv2D(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, act_layer=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.weight.data[:in_channel, :self.active_out_channel, :, :])
        sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
        # act parameters to be added ~
        return sub_layer


class IBasicBottleneckDenseBlock(BasicBlock):
    def __init__(self, in_channels_list, out_channels_list,
                 kernel_size, stride, dilation=1,
                 output_padding=1,
                 bias=True, Tconv=False):
        super(IBasicBottleneckDenseBlock, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.dilation = dilation

        # Basic 2D convolution
        if Tconv:
            self.conv = DSTransposeConv2d(in_channels_list,
                                          out_channels_list,
                                          kernel_size=kernel_size,
                                          stride=(stride, stride),
                                          bias=bias,
                                          output_padding=(output_padding, output_padding))
            self.tconv = True
        else:
            self.conv = DSConv2d(in_channels_list,
                                 out_channels_list,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation=(dilation, dilation),
                                 bias=bias)
            self.tconv = False


        self.block_1 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.block_2 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.block_3 = nn.ModuleList([
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(3, 3),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d(out_channels_list,
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias),
            DSConv2d([channel * 3 for channel in out_channels_list],
                     out_channels_list,
                     kernel_size=(1, 1),
                     stride=(1, 1),
                     dilation=(1, 1),
                     bias=bias,
                     cat_factor=3),
        ]
        )

        self.global_fusion = DSConv2d([channel * 3 for channel in out_channels_list],
                                      out_channels_list,
                                      kernel_size=(1, 1),
                                      stride=(1, 1),
                                      dilation=(1, 1),
                                      bias=bias,
                                      cat_factor=3)

        self.initialize()
        self.active_out_channel = out_channels_list[-1]  # research_result~

    def forward(self, x):
        x = self.conv(x)
        main = x
        self.active_out_channel = self.conv.active_out_channel
        self.set_active_channels()

        x_1 = F.relu(self.block_1[0](x), inplace=True)
        x_2 = F.relu(self.block_1[1](x_1), inplace=True)
        x_3 = F.relu(self.block_1[2](x_2), inplace=True)
        identity_1 = self.block_1[3](torch.cat([x_1, x_2, x_3], dim=1))
        x = x + identity_1

        x_1 = F.relu(self.block_2[0](x), inplace=True)
        x_2 = F.relu(self.block_2[1](x_1), inplace=True)
        x_3 = F.relu(self.block_2[2](x_2), inplace=True)
        identity_2 = self.block_2[3](torch.cat([x_1, x_2, x_3], dim=1))
        x = x + identity_2

        x_1 = F.relu(self.block_3[0](x), inplace=True)
        x_2 = F.relu(self.block_3[1](x_1), inplace=True)
        x_3 = F.relu(self.block_3[2](x_2), inplace=True)
        identity_3 = self.block_3[3](torch.cat([x_1, x_2, x_3], dim=1))
        #
        x = main + self.global_fusion(torch.cat([identity_1, identity_2, identity_3], dim=1))
        return x

    @property
    def config(self):
        return {
            'name': IBasicBottleneckDenseBlock.__name__,
            'in_channel_list': self.in_channels_list,
            'out_channel_list': self.out_channels_list,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'act_func': None,
            'tconv': self.tconv
        }

    @staticmethod
    def build_from_config(config):
        return IBasicBottleneckDenseBlock(**config)

    def get_active_subnet(self, in_channel, preserve_weight=True):
        self.active_out_channel = self.out_channels_list[self.channel_choice]
        sub_layer = IBasicConv2D(
            in_channel, self.active_out_channel, self.kernel_size, self.stride, self.dilation, act_layer=self.act_func
        )
        sub_layer = sub_layer.to(get_net_device(self))

        if not preserve_weight:
            return sub_layer

        sub_layer.conv.weight.data.copy_(self.conv.weight.data[:self.active_out_channel, :in_channel, :, :])
        sub_layer.conv.bias.data.copy_(self.conv.bias.data[:self.active_out_channel])
        # act parameters to be added ~
        return sub_layer

    def set_active_channels(self):
        for n, m in self.named_modules():
            set_exist_attr(m, 'active_out_channel', self.active_out_channel)



if __name__ == "__main__":
    pass
