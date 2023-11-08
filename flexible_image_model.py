# -*- coding: utf-8 -*-
from Base.base_model import GainChannelAutoAll
from flexible_image_module import *
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from flexible_rate_module import Gain_Module
from compressai.models.utils import conv

class GainCA(GainChannelAutoAll):
    def __init__(self, in_ch=3, N=192, M=320, out_ch=3, se_ratio=0.8, lower_channel=8,
                 channel_gap=24, complexity_point=4, rate_point=5):
        super().__init__()
        self.num_slices = 10
        self.max_support_slices = 5


        self.N = int(N)
        self.M = int(M)
        self.complexity_point = complexity_point
        self.rate_point = rate_point


        N_channel_list = list(range(48, N + 1, 18))
        M_channel_list = list(range(80, M + 1, 30))
        M_channel_list_Scare = list(range(int(64),
                                          int(256+1), int(24)))

        slice_depth = self.M // self.num_slices
        if slice_depth * self.num_slices != self.M:
            raise ValueError(f"Slice do not evenly divide latent depth ({self.M}/{self.num_slices})")

        print("N_channel_num: ", len(N_channel_list), " channel : ", N_channel_list)
        print("M_channel_num: ", len(M_channel_list), " channel : ", M_channel_list)
        print("M_channel_num: ", len(M_channel_list_Scare), " channel : ", M_channel_list_Scare)

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)
        self.gain_unit = Gain_Module(n=rate_point, N=M)
        self.inv_gain_unit = Gain_Module(n=rate_point, N=M)
        self.prior_gain_unit = Gain_Module(n=rate_point, N=N)
        self.prior_inv_gain_unit = Gain_Module(n=rate_point, N=N)

        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False
        self.complexity_level = None

        self.g_a = g_a_dense(
            in_ch=in_ch, N=N, M=M, out_ch=out_ch, se_ratio=se_ratio,
            complexity_point=complexity_point, rate_point=rate_point,
            N_channel_list=N_channel_list, M_channel_list=M_channel_list,
            M_channel_list_Scare=M_channel_list_Scare
        )

        self.g_s = g_s_dense(
            in_ch=in_ch, N=N, M=M, out_ch=out_ch, se_ratio=se_ratio,
            complexity_point=complexity_point, rate_point=rate_point,
            N_channel_list=N_channel_list, M_channel_list=M_channel_list,
            M_channel_list_Scare=M_channel_list_Scare
        )

        self.h_a = mh_a_all(
            in_ch=in_ch, N=N, M=M, out_ch=out_ch, se_ratio=se_ratio,
            complexity_point=complexity_point, rate_point=rate_point,
            N_channel_list=N_channel_list, M_channel_list=M_channel_list,
            M_channel_list_Scare=M_channel_list_Scare
        )

        self.h_scale_s = mh_s_all(
            in_ch=in_ch, N=N, M=M, out_ch=out_ch, se_ratio=se_ratio,
            complexity_point=complexity_point, rate_point=rate_point,
            N_channel_list=N_channel_list, M_channel_list=M_channel_list,
            M_channel_list_Scare=M_channel_list_Scare
        )

        self.h_mean_s = mh_s_all(
            in_ch=in_ch, N=N, M=M, out_ch=out_ch, se_ratio=se_ratio,
            complexity_point=complexity_point, rate_point=rate_point,
            N_channel_list=N_channel_list, M_channel_list=M_channel_list,
            M_channel_list_Scare=M_channel_list_Scare
        )

        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i, self.max_support_slices), 224, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                conv(M + slice_depth * min(i + 1, self.max_support_slices + 1), 224, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=3),
                nn.ReLU(inplace=True),
                conv(128, slice_depth, stride=1, kernel_size=3)
            ) for i in range(self.num_slices)
        )
        self.cfg_candidates = {
            'g_a': {
                'layer_num': self.g_a.layer_num,
                'c': self.g_a.width_list},
            'g_s': {
                'layer_num': self.g_s.layer_num,
                'c': self.g_s.width_list},
            'h_a': {
                'layer_num': self.h_a.layer_num,
                'c': self.h_a.width_list},
            'h_mean_s': {
                'layer_num': self.h_mean_s.layer_num,
                'c': self.h_mean_s.width_list},
            'h_scale_s': {
                'layer_num': self.h_scale_s.layer_num,
                'c': self.h_scale_s.width_list},
        }

if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    h, w = 768, 512
    # h, w = 1920, 1088

    model = GainCA()
    for key, value in model.cfg_candidates.items():
        print(f"model : {key},  attribute: {value}")
    bias = True
    g_a_parameters = sum(p.numel() for p in model.g_a.parameters() if p.requires_grad)
    g_s_parameters = sum(p.numel() for p in model.g_s.parameters() if p.requires_grad)
    h_s_parameters = sum(p.numel() for p in model.h_mean_s.parameters() if p.requires_grad) + \
                     sum(p.numel() for p in model.h_scale_s.parameters() if p.requires_grad)

    context_model_parameters = sum(p.numel() for p in model.cc_mean_transforms.parameters() if p.requires_grad) + \
    sum(p.numel() for p in model.cc_scale_transforms.parameters() if p.requires_grad) + \
    sum(p.numel() for p in model.lrp_transforms.parameters() if p.requires_grad)

    total_parameters = g_s_parameters + h_s_parameters + context_model_parameters

    print(
        f'g_s Total Parameters = {round(g_s_parameters / total_parameters * 100)} %')
    print(
        f'h_s Total Parameters = {round(h_s_parameters / total_parameters * 100)} %')
    print(
        f'context_model Parameters = {round(context_model_parameters / total_parameters * 100)} %')
    print(f'g_a Parameters = {g_a_parameters}')
    print(f"Total Parameters: {total_parameters}")

    bias = False

    x = torch.rand((3, 3, h, w))
    model.sample_active_subnet(mode='smallest')
    curr_cfg = {'g_a': [192, 192, 192, 320], 'h_a': [], 'g_s': [192, 192, 192, 3],
                'h_mean_s': [int(80), int(165), 320],
                'h_scale_s': [int(80), int(165), 320]}

    model.set_active_subnet(curr_cfg)

    g_s_parameters_count = model.compute_active_subnet_parameters((h, w), g_s=True, h_s=False, bias=True)
    h_s_parameters_count = model.compute_active_subnet_parameters((h, w), g_s=False, h_s=True, bias=True)

    width = model.get_active_subnet_settings()

    print(g_s_parameters_count, g_s_parameters)
    print(h_s_parameters_count, h_s_parameters)
    print(round((h_s_parameters - h_s_parameters_count) / total_parameters * 100, 3))

    factor = 0.25
    channel = 192*0.25
    # curr_cfg = {'g_a': [192, 192, 192, 320], 'h_a': [], 'g_s': [channel, channel, channel, 3],
    #             'h_s': [int(320 * factor), int(480 * factor), 640],
    #             'entropy': [int(1280 * factor), int(1066 * factor), 853]}

    curr_cfg = {'g_a': [192, 192, 192, 320], 'h_a': [], 'g_s': [channel, channel, channel, 3],
                'h_mean_s': [int(192 * factor), int(256 * factor), 320],
                'h_scale_s': [int(192 * factor), int(256 * factor), 320]}
    model.set_active_subnet(curr_cfg)
    curr_flops = model.compute_active_subnet_flops((h, w), g_s=True, bias=bias)

    model.set_rate_level(0)
    total = 0
    flops_g_s = model.compute_active_subnet_flops((h, w), g_s=True, bias=bias)
    total += flops_g_s
    flops_h_s = model.compute_active_subnet_flops((h, w), g_s=False, h_s=True, bias=bias)
    total += flops_h_s
    flops_entropy = model.compute_active_subnet_flops((h, w), g_s=False, h_s=True, context_model=True, bias=bias)
    print('Context Model + h_s', (flops_entropy) / h / w / 1000000)
    exit()
    total += flops_entropy
    flops_entropy = model.compute_active_subnet_flops((h, w), g_s=True, h_s=False, context_model=True, bias=bias)

    model.sample_active_subnet(mode='largest')
    largest_flops = model.compute_active_subnet_flops((h, w), g_s=True, h_s=False, context_model=False, bias=bias)
    g_a_largest_flops = model.compute_active_subnet_flops((h, w), g_a=True, g_s=False, h_s=False, context_model=False, bias=bias)
    print(largest_flops, g_a_largest_flops)
    model.sample_active_subnet(mode='smallest')
    smallest_flops = model.compute_active_subnet_flops((h, w), g_s=True, h_s=False, context_model=False, bias=bias)

    smallest_parameters = model.compute_active_subnet_parameters((h, w), g_s=True, h_s=False, context_model=False, bias=bias)
    print(smallest_flops / largest_flops * 100)
    print("g_s 768*512", smallest_flops)
    print("g_a 768*512", g_a_largest_flops)

    print("smallest_parameters:", smallest_parameters)
    print((320*192*h*w/8/8*25 + 192*192*h*w/4/4*25 + 192*192*h*w/2/2*25 + 192*3*h*w*25) / 1000000000)
    print(smallest_flops * 4 / 1000000000)

    print("g_s 768*512 NIPS 2018 --192", (320*192*h*w/8/8*25 + 192*192*h*w/4/4*25 + 192*192*h*w/2/2*25 + 192*3*h*w*25))
    print("g_a 768*512 NIPS 2018 --192", (3*192*h*w/2/2*25 + 192*192*h*w/4/4*25 + 192*192*h*w/8/8*25 + 192*320*h*w/16/16*25))

    # 32086425600  84934656000
    print("g_s 768*512 NIPS 2018 --128", (3774873600 + 10066329600 + 40265318400 + 3774873600) / smallest_flops)

    print(f'fatcor:{factor}', curr_flops / largest_flops * 100)

    print(f"g_s floaps: {flops_g_s / total * 100}")
    print(f"h_s floaps: {flops_h_s / total * 100}")
    print(f"entropy_parameters floaps: {flops_entropy / total * 100}")


    model.sample_active_subnet(mode='largest')
    # max_complexity_decoder_all = model.compute_active_subnet_flops(
    #     image_size=(h, w),
    #     g_s=True, h_s=True, bias=False, context_model=True)

    max_complexity_encoder = model.compute_active_subnet_flops(image_size=(h, w),
                                                               g_s=False, h_s=True, bias=False, context_model=True,
                                                               g_a=True)
    print('       ~')
    max_complexity_decoder = model.compute_active_subnet_flops(image_size=(h, w),
                                                               g_s=True)
    # print(max_complexity_decoder_all)
    print(max_complexity_encoder/1e9)
    print(max_complexity_decoder/1e9)

    exit()
