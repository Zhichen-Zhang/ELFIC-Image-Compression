# -*- coding: utf-8 -*-
from dyn_slim_blocks import IBasicConv2D, IBasicTransposeConv2D, IBasicBottleneckDenseBlock
import torch.nn as nn

class g_a_dense(nn.Module):
    def __init__(self, in_ch=3, N=128, M=192, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.g_a = nn.Sequential(
            IBasicBottleneckDenseBlock(in_channels_list=[3], out_channels_list=[N],
                                       kernel_size=(5, 5), stride=2, Tconv=False),

            IBasicBottleneckDenseBlock(in_channels_list=[N], out_channels_list=[N],
                                       kernel_size=(5, 5), stride=2, Tconv=False),

            IBasicBottleneckDenseBlock(in_channels_list=[N], out_channels_list=[N],
                                       kernel_size=(5, 5), stride=2, Tconv=False),

            IBasicBottleneckDenseBlock(in_channels_list=[N], out_channels_list=[M],
                                       kernel_size=(5, 5), stride=2, Tconv=False),
        )
        self.width_list = []
        self.layer_num = 0
        for module in self.g_a:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicBottleneckDenseBlock):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, x):
        return self.g_a(x)


class g_s_dense(nn.Module):
    def __init__(self, in_ch=3, N=128, M=192, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.g_s = nn.ModuleList([
            IBasicBottleneckDenseBlock(in_channels_list=[M], out_channels_list=N_channel_list,
                                       kernel_size=(5, 5), stride=2, Tconv=True),

            IBasicBottleneckDenseBlock(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                       kernel_size=(5, 5), stride=2, Tconv=True),

            IBasicBottleneckDenseBlock(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                                       kernel_size=(5, 5), stride=2, Tconv=True),
            IBasicTransposeConv2D(in_channels_list=N_channel_list, out_channels_list=[out_ch],
                                  kernel_size=(5, 5), stride=2,
                                  act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.g_s:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicBottleneckDenseBlock):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, x_hat):
        for module in self.g_s:
            x_hat = module(x_hat)
        return x_hat


class h_s_all(nn.Module):
    def __init__(self, in_ch=3, N=128, M=192, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False

        self.h_s = nn.ModuleList([
            IBasicTransposeConv2D(in_channels_list=[N], out_channels_list=M_channel_list,
                                  kernel_size=(5, 5), stride=2,
                                  act_layer="Relu"),
            IBasicTransposeConv2D(in_channels_list=M_channel_list, out_channels_list=M_channel_list_Scare,
                                  kernel_size=(5, 5), stride=2,
                                  act_layer="Relu"),
            IBasicConv2D(M_channel_list_Scare, [M * 2], kernel_size=(3, 3), stride=1,
                         act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.h_s:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, gaussian_params):
        for module in self.h_s:  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
            gaussian_params = module(gaussian_params)
        return gaussian_params


class mh_s_all(nn.Module):
    def __init__(self, in_ch=3, N=192, M=320, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False

        self.h_s = nn.ModuleList([
            IBasicTransposeConv2D(in_channels_list=[N], out_channels_list=N_channel_list,
                                  kernel_size=(5, 5), stride=2,
                                  act_layer="Relu"),
            IBasicTransposeConv2D(in_channels_list=N_channel_list, out_channels_list=M_channel_list_Scare,
                                  kernel_size=(5, 5), stride=2,
                                  act_layer="Relu"),
            IBasicConv2D(M_channel_list_Scare, [M], kernel_size=(3, 3), stride=1,
                         act_layer=None),
        ])
        self.width_list = []
        self.layer_num = 0
        for module in self.h_s:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, gaussian_params):
        for module in self.h_s:  # [b, 128, w//64, h//64] -> [b, 256, w//16, h//16]
            gaussian_params = module(gaussian_params)
        return gaussian_params


class h_a_all(nn.Module):
    def __init__(self, in_ch=3, N=128, M=192, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.h_a = nn.Sequential(
            IBasicConv2D(in_channels_list=[M], out_channels_list=N_channel_list,
                         kernel_size=(3, 3), stride=(1, 1), act_layer='Relu'),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=N_channel_list,
                         kernel_size=(5, 5), stride=(2, 2), act_layer='Relu'),
            IBasicConv2D(in_channels_list=N_channel_list, out_channels_list=[N],
                         kernel_size=(5, 5), stride=(2, 2), act_layer=None),
        )

        self.width_list = []
        self.layer_num = 0
        for module in self.h_a:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, z):
        return self.h_a(z)


class mh_a_all(nn.Module):
    def __init__(self, in_ch=3, N=128, M=192, out_ch=3, se_ratio=0.8,
                 complexity_point=4, rate_point=1,
                 N_channel_list=None, M_channel_list=None, M_channel_list_Scare=None):
        super().__init__()
        self.N = int(N)
        self.M = int(M)

        self.h_a = nn.Sequential(
            IBasicConv2D(in_channels_list=[M], out_channels_list=[320],
                         kernel_size=(3, 3), stride=(1, 1), act_layer='Relu'),
            IBasicConv2D(in_channels_list=[320], out_channels_list=[256],
                         kernel_size=(5, 5), stride=(2, 2), act_layer='Relu'),
            IBasicConv2D(in_channels_list=[256], out_channels_list=[192],
                         kernel_size=(5, 5), stride=(2, 2), act_layer=None),
        )

        self.width_list = []
        self.layer_num = 0
        for module in self.h_a:
            if isinstance(module, IBasicTransposeConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1
            elif isinstance(module, IBasicConv2D):
                self.width_list.append(module.out_channels_list)
                self.layer_num += 1

    def forward(self, z):
        return self.h_a(z)

