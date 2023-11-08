# -*- coding: utf-8 -*-
import math
import torch
import random
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.utils import update_registered_buffers
from compressai.ops import ste_round
from dyn_slim_blocks import *
from Base.base_utils import cal_psnr

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class GainChannelAutoAll(nn.Module):
    def __init__(self, N=192, M=320):
        super().__init__()
        self.num_slices = 10
        self.max_support_slices = 5

        self.g_a = None
        self.g_s = None
        self.h_a = None

        self.h_mean_s = None
        self.h_scale_s = None
        self.cc_mean_transforms = None
        self.cc_scale_transforms = None
        self.lrp_transforms = None

        self.gain_unit = None
        self.inv_gain_unit = None
        self.prior_gain_unit = None
        self.prior_inv_gain_unit = None

        self.rate_level, self.rate_interpolation_coefficient = None, None
        self.prior_rate_level, self.prior_rate_interpolation_coefficient = None, None
        self.isInterpolation = False
        self.complexity_level = None

    def forward(self, x):
        y = self.g_a(x)
        y_gained = self.gain_unit(y, self.rate_level, self.rate_interpolation_coefficient)
        z = self.h_a(y_gained)
        z_gained = self.prior_gain_unit(z, self.rate_level, self.rate_interpolation_coefficient)
        y_shape = y.shape[2:]
        _, z_likelihoods = self.entropy_bottleneck(z_gained)

        # Use rounding (instead of uniform noise) to modify z before passing it
        # to the hyper-synthesis transforms. Note that quantize() overrides the
        # gradient to create a straight-through estimator.
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z_gained - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                    self.rate_interpolation_coefficient)
        latent_scales = self.h_scale_s(z_hat_inv_gained)
        latent_means = self.h_mean_s(z_hat_inv_gained)

        y_slices = y_gained.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []
        y_scales = []
        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp
            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        y_hat_inv_gained = self.inv_gain_unit(y_hat, self.rate_level, self.rate_interpolation_coefficient)
        x_hat = self.g_s(y_hat_inv_gained)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def evaluate(self, x, all_cfg, min_rate_level, max_rate_level, rate_interpolation_coefficient=0):
        y = self.g_a(x)  # equal y = self.g_a(x)
        y_shape = y.shape[2:]
        y_gained_list = [self.gain_unit(y, rate_level, rate_interpolation_coefficient) for rate_level in
                         range(min_rate_level, max_rate_level)]
        z_list = [self.h_a(y_gained) for y_gained in y_gained_list]

        z_gained_list = [
            self.prior_gain_unit(z_list[rate_level - min_rate_level], rate_level, rate_interpolation_coefficient)
            for rate_level in range(min_rate_level, max_rate_level)]
        z_hat_list = []
        z_likelihoods_list = []
        for z_gained in z_gained_list:
            _, z_likelihoods = self.entropy_bottleneck(z_gained)
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z_gained - z_offset
            z_hat = ste_round(z_tmp) + z_offset
            z_hat_list.append(z_hat)
            z_likelihoods_list.append(z_likelihoods)
        latent_scales_inv_gained_list = []
        latent_means_inv_gained_list = []
        gaussian_params_inv_gained_list = []
        for rate_level in range(min_rate_level, max_rate_level):
            gaussian_params_inv_gained = self.prior_inv_gain_unit(z_hat_list[rate_level - min_rate_level], rate_level,
                                                                  rate_interpolation_coefficient)
            latent_scales = self.h_scale_s(gaussian_params_inv_gained)
            latent_means = self.h_mean_s(gaussian_params_inv_gained)
            latent_scales_inv_gained_list.append(latent_scales)
            latent_means_inv_gained_list.append(latent_means)

        if all_cfg is not None:
            cfg_x_hat = []
            cfg_z_likelihoods = []
            cfg_y_likelihoods = []
            for cfg in all_cfg:
                self.set_active_subnet(cfg)
                y_hat_list = []
                y_likelihoods_list = []
                for i in range(len(y_gained_list)):
                    y_slices = y_gained_list[i].chunk(self.num_slices, 1)
                    y_hat_slices = []
                    y_likelihood = []

                    for slice_index, y_slice in enumerate(y_slices):
                        support_slices = (
                            y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                        mean_support = torch.cat([latent_means_inv_gained_list[i]] + support_slices, dim=1)
                        mu = self.cc_mean_transforms[slice_index](mean_support)
                        mu = mu[:, :, :y_shape[0], :y_shape[1]]

                        scale_support = torch.cat([latent_scales_inv_gained_list[i]] + support_slices, dim=1)
                        scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
                        scale = scale[:, :, :y_shape[0], :y_shape[1]]

                        _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                        y_likelihood.append(y_slice_likelihood)
                        y_hat_slice = ste_round(y_slice - mu) + mu

                        lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                        lrp = self.lrp_transforms[slice_index](lrp_support)
                        lrp = 0.5 * torch.tanh(lrp)
                        y_hat_slice += lrp

                        y_hat_slices.append(y_hat_slice)

                    y_hat_list.append(torch.cat(y_hat_slices, dim=1))
                    y_likelihoods = torch.cat(y_likelihood, dim=1)
                    y_likelihoods_list.append(y_likelihoods)

                x_hat_inv_gained_list = [self.inv_gain_unit(y_hat_list[rate_level - min_rate_level],
                                                            rate_level, rate_interpolation_coefficient)
                                         for rate_level in range(min_rate_level, max_rate_level)]
                x_hat_list = [self.g_s(x_hat_inv_gained) for x_hat_inv_gained in x_hat_inv_gained_list]
                cfg_x_hat.append(x_hat_list)
                cfg_y_likelihoods.append(y_likelihoods_list)
                cfg_z_likelihoods.append(z_likelihoods_list)
            return {
                "x_hat": cfg_x_hat,
                "likelihoods": {"y": cfg_y_likelihoods, "z": cfg_z_likelihoods},
            }
        else:
            y_likelihoods_list = []
            y_hat_list = []

            for i in range(len(y_gained_list)):
                y_slices = y_gained_list[i].chunk(self.num_slices, 1)
                y_hat_slices = []
                y_likelihood = []

                for slice_index, y_slice in enumerate(y_slices):
                    support_slices = (
                        y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                    mean_support = torch.cat([latent_means_inv_gained_list[i]] + support_slices, dim=1)
                    mu = self.cc_mean_transforms[slice_index](mean_support)
                    mu = mu[:, :, :y_shape[0], :y_shape[1]]

                    scale_support = torch.cat([latent_scales_inv_gained_list[i]] + support_slices, dim=1)
                    scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
                    scale = scale[:, :, :y_shape[0], :y_shape[1]]

                    _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
                    y_likelihood.append(y_slice_likelihood)
                    y_hat_slice = ste_round(y_slice - mu) + mu

                    lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                    lrp = self.lrp_transforms[slice_index](lrp_support)
                    lrp = 0.5 * torch.tanh(lrp)
                    y_hat_slice += lrp

                    y_hat_slices.append(y_hat_slice)

                y_hat_list.append(torch.cat(y_hat_slices, dim=1))
                y_likelihoods = torch.cat(y_likelihood, dim=1)
                y_likelihoods_list.append(y_likelihoods)

            x_hat_inv_gained_list = [self.inv_gain_unit(y_hat_list[rate_level - min_rate_level],
                                                        rate_level, rate_interpolation_coefficient)
                                     for rate_level in range(min_rate_level, max_rate_level)]
            x_hat_list = [self.g_s(x_hat_inv_gained) for x_hat_inv_gained in x_hat_inv_gained_list]
            return {
                "x_hat": x_hat_list,
                "likelihoods": {"y": y_likelihoods_list, "z": z_likelihoods_list},
            }

    def evaluate_compress(self, x, all_cfg, min_rate_level, max_rate_level, rate_interpolation_coefficient=0):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        y_gained_list = [self.gain_unit(y, rate_level, rate_interpolation_coefficient) for rate_level in
                         range(min_rate_level, max_rate_level)]
        z_list = [self.h_a(y_gained) for y_gained in y_gained_list]

        z_gained_list = [
            self.prior_gain_unit(z_list[rate_level - min_rate_level], rate_level, rate_interpolation_coefficient)
            for rate_level in range(min_rate_level, max_rate_level)]

        z_hat_list = []
        z_strings_list = []
        z_likelihoods_list = []
        for z_gained in z_gained_list:
            z_bpp = 0
            z_strings = self.entropy_bottleneck.compress(z_gained)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z_gained.size()[-2:])
            z_hat_list.append(z_hat)
            for i in range(len(z_strings)):
                z_bpp += len(z_strings[i])
            z_likelihoods_list.append(z_bpp)
            z_strings_list.append(z_strings)

        latent_scales_inv_gained_list = []
        latent_means_inv_gained_list = []

        self.set_active_subnet(all_cfg[0])
        for rate_level in range(min_rate_level, max_rate_level):
            gaussian_params_inv_gained = self.prior_inv_gain_unit(z_hat_list[rate_level - min_rate_level], rate_level,
                                                                  rate_interpolation_coefficient)
            latent_scales = self.h_scale_s(gaussian_params_inv_gained)
            latent_means = self.h_mean_s(gaussian_params_inv_gained)
            latent_scales_inv_gained_list.append(latent_scales)
            latent_means_inv_gained_list.append(latent_means)

        y_likelihoods_list = []
        y_strings_list = []
        for i in range(len(y_gained_list)):
            y_slices = y_gained_list[i].chunk(self.num_slices, 1)
            y_bpp = 0
            y_hat_slices = []
            y_scales = []
            y_means = []

            cdf = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

            encoder = BufferedRansEncoder()
            symbols_list = []
            indexes_list = []
            y_strings = []

            for slice_index, y_slice in enumerate(y_slices):
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                mean_support = torch.cat([latent_means_inv_gained_list[i]] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales_inv_gained_list[i]] + support_slices, dim=1)
                scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                index = self.gaussian_conditional.build_indexes(scale)
                y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
                y_hat_slice = y_q_slice + mu

                symbols_list.extend(y_q_slice.reshape(-1).tolist())
                indexes_list.extend(index.reshape(-1).tolist())

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp

                y_hat_slices.append(y_hat_slice)
                y_scales.append(scale)
                y_means.append(mu)

            encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
            y_string = encoder.flush()
            y_strings.append(y_string)
            for j in range(len(y_strings)):
                y_bpp += len(y_strings[j])
            y_likelihoods_list.append(y_bpp)
            y_strings_list.append(y_strings)

        y_hat_list = []
        for i, y_string in enumerate(y_strings_list):
            y_hat_slices = []
            cdf = self.gaussian_conditional.quantized_cdf.tolist()
            cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
            offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

            decoder = RansDecoder()
            decoder.set_stream(y_string[0])

            for slice_index in range(self.num_slices):
                support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
                mean_support = torch.cat([latent_means_inv_gained_list[i]] + support_slices, dim=1)
                mu = self.cc_mean_transforms[slice_index](mean_support)
                mu = mu[:, :, :y_shape[0], :y_shape[1]]

                scale_support = torch.cat([latent_scales_inv_gained_list[i]] + support_slices, dim=1)
                scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
                scale = scale[:, :, :y_shape[0], :y_shape[1]]

                index = self.gaussian_conditional.build_indexes(scale)

                rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
                rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
                y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

                lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
                lrp = self.lrp_transforms[slice_index](lrp_support)
                lrp = 0.5 * torch.tanh(lrp)
                y_hat_slice += lrp
                y_hat_slices.append(y_hat_slice)

            y_hat = torch.cat(y_hat_slices, dim=1)
            y_hat_list.append(y_hat)

        if all_cfg is not None:
            cfg_x_hat = []
            cfg_z_likelihoods = []
            cfg_y_likelihoods = []
            for cfg in all_cfg:
                self.set_active_subnet(cfg)
                x_hat_inv_gained_list = [self.inv_gain_unit(y_hat_list[rate_level - min_rate_level],
                                                            rate_level, rate_interpolation_coefficient)
                                         for rate_level in range(min_rate_level, max_rate_level)]
                x_hat_list = [self.g_s(x_hat_inv_gained).clamp_(0, 1) for x_hat_inv_gained in x_hat_inv_gained_list]
                cfg_x_hat.append(x_hat_list)
                cfg_y_likelihoods.append(y_likelihoods_list)
                cfg_z_likelihoods.append(z_likelihoods_list)
            return {
                "x_hat": cfg_x_hat,
                "likelihoods": {"y": cfg_y_likelihoods, "z": cfg_z_likelihoods},
            }
        else:
            x_hat_inv_gained_list = [self.inv_gain_unit(y_hat_list[rate_level - min_rate_level],
                                                        rate_level, rate_interpolation_coefficient)
                                     for rate_level in range(min_rate_level, max_rate_level)]
            x_hat_list = [self.g_s(x_hat_inv_gained).clamp_(0, 1) for x_hat_inv_gained in x_hat_inv_gained_list]
            return {
                "x_hat": x_hat_list,
                "likelihoods": {"y": y_likelihoods_list, "z": z_likelihoods_list},
            }

    def compress(self, x):
        y = self.g_a(x)
        y_gained = self.gain_unit(y, self.rate_level, self.rate_interpolation_coefficient, self.isInterpolation)
        z = self.h_a(y_gained)
        z_gained = self.prior_gain_unit(z, self.rate_level, self.rate_interpolation_coefficient, self.isInterpolation)
        y_shape = y_gained.shape[2:]

        z_strings = self.entropy_bottleneck.compress(z_gained)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                    self.rate_interpolation_coefficient, self.isInterpolation)
        latent_scales = self.h_scale_s(z_hat_inv_gained)
        latent_means = self.h_mean_s(z_hat_inv_gained)

        y_slices = y_gained.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)

        z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, self.rate_level,
                                                    self.rate_interpolation_coefficient, self.isInterpolation)
        latent_scales = self.h_scale_s(z_hat_inv_gained)
        latent_means = self.h_mean_s(z_hat_inv_gained)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat_inv_gained = self.inv_gain_unit(y_hat, self.rate_level, self.rate_interpolation_coefficient, self.isInterpolation)
        x_hat = self.g_s(y_hat_inv_gained).clamp_(0, 1)
        return {"x_hat": x_hat}

    def IDCA_compress(self, x, all_cfg, rate_level, rate_interpolation_coefficient=0):
        y = self.g_a(x)
        y_shape = y.shape[2:]
        y_gained = self.gain_unit(y, rate_level, rate_interpolation_coefficient)
        z = self.h_a(y_gained)
        z_gained = self.prior_gain_unit(z, rate_level, rate_interpolation_coefficient)

        z_strings = self.entropy_bottleneck.compress(z_gained)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z_gained.size()[-2:])
        z_bpp = 0
        for i in range(len(z_strings)):
            z_bpp += len(z_strings[i])

        latent_scales_inv_gained_list = []
        latent_means_inv_gained_list = []

        self.set_active_subnet(all_cfg[0])
        gaussian_params_inv_gained = self.prior_inv_gain_unit(z_hat, rate_level,
                                                              rate_interpolation_coefficient)
        latent_scales = self.h_scale_s(gaussian_params_inv_gained)
        latent_means = self.h_mean_s(gaussian_params_inv_gained)
        latent_scales_inv_gained_list.append(latent_scales)
        latent_means_inv_gained_list.append(latent_means)

        y_slices = y_gained.chunk(self.num_slices, 1)
        y_bpp = 0
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means_inv_gained_list[i]] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales_inv_gained_list[i]] + support_slices, dim=1)
            scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        for j in range(len(y_strings)):
            y_bpp += len(y_strings[j])

        y_hat = torch.cat(y_hat_slices, dim=1)
        if all_cfg is not None:
            cfg_x_hat_psnr = []
            for cfg in all_cfg:
                self.set_active_subnet(cfg)
                x_hat_inv_gained = self.inv_gain_unit(y_hat, rate_level, rate_interpolation_coefficient)
                x_hat = self.g_s(x_hat_inv_gained).clamp_(0, 1)
                psnr = cal_psnr(x_hat, x) # or SSIM
                cfg_x_hat_psnr.append(psnr)
            return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                        "quality_list": cfg_x_hat_psnr}

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "quality_list": None}

    def IDCA_decompress(self, strings, shape, rate_level, rate_interpolation_coefficient=0,
                        all_cfg=None, quality_list=None, lamdaC=None):
        cfg = self.getcfg(all_cfg, lamdaC, quality_list)
        self.set_active_subnet(cfg)
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        z_hat_inv_gained = self.prior_inv_gain_unit(z_hat, rate_level, rate_interpolation_coefficient, self.isInterpolation)
        latent_scales = self.h_scale_s(z_hat_inv_gained)
        latent_means = self.h_mean_s(z_hat_inv_gained)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = torch.exp(self.cc_scale_transforms[slice_index](scale_support))
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat_inv_gained = self.inv_gain_unit(y_hat, rate_level, rate_interpolation_coefficient, self.isInterpolation)
        x_hat = self.g_s(y_hat_inv_gained).clamp_(0, 1)
        return {"x_hat": x_hat}

    def getcfg(self, all_cfg, lamdaC, quality_list):
        ans = -1
        max = -10000
        for i, cur_cfg in enumerate(all_cfg):
            cost = quality_list[i] - float(cur_cfg['complexity']) * lamdaC / 100
            if(cost > max):
                max = cost
                ans = i
        return all_cfg[ans]

    def aux_loss(self):
        aux_loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return aux_loss

    def update(self, scale_table=None, force=False):
        self.entropy_bottleneck.update(force=force)

        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        return updated

    def load_state_dict(self, state_dict, pretrain=False):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # Dynamically update the entropy bottleneck buffers related to the CDFs
        update_registered_buffers(
            self.entropy_bottleneck,
            "entropy_bottleneck",
            ["_quantized_cdf", "_offset", "_cdf_length"],
            state_dict,
        )
        if pretrain:
            super().load_state_dict(state_dict, strict=False)
        else:
            super().load_state_dict(state_dict)

    def set_rate_level(self, index, interpolation_coefficient=1.0, isInterpolation=False):
        self.rate_level = index
        self.prior_rate_level = index
        self.rate_interpolation_coefficient = interpolation_coefficient
        self.prior_rate_interpolation_coefficient = interpolation_coefficient
        self.isInterpolation = isInterpolation

    def set_active_subnet(self, cfg, **kwargs):
        for layer_index, layer in enumerate(self.g_s.g_s):
            layer.conv.active_out_channel = cfg['g_s'][layer_index]

        for layer_index, layer in enumerate(self.h_mean_s.h_s):
            layer.conv.active_out_channel = cfg['h_mean_s'][layer_index]

        for layer_index, layer in enumerate(self.h_scale_s.h_s):
            layer.conv.active_out_channel = cfg['h_scale_s'][layer_index]

    def get_active_subnet_settings(self):
        width = {}
        for k in ['g_a', 'h_a', 'g_s', 'h_mean_s', 'h_scale_s']:
            width[k] = []

        for layer_index, layer in enumerate(self.g_s.g_s):
            width['g_s'].append(layer.conv.active_out_channel)

        for layer_index, layer in enumerate(self.h_mean_s.h_s):
            width['h_mean_s'].append(layer.conv.active_out_channel)

        for layer_index, layer in enumerate(self.h_scale_s.h_s):
            width['h_scale_s'].append(layer.conv.active_out_channel)

        return {"width": width}

    def sample_active_subnet(self, mode='largest', compute_flops=False):
        assert mode in ['largest', 'random', 'smallest', 'uniform', 'random_uniform']
        if mode == 'random':
            cfg = self._sample_active_subnet(min_net=False, max_net=False)
        elif mode == 'largest':
            cfg = self._sample_active_subnet(max_net=True)
        elif mode == 'smallest':
            cfg = self._sample_active_subnet(min_net=True)
        elif mode == 'uniform':
            cfg = self._sample_active_subnet(uniform=True)
        elif mode == 'random_uniform':
            cfg = self._sample_active_subnet(random_uniform=True)

        if compute_flops:
            cfg['complexity'] = self.compute_active_subnet_flops()
        return cfg

    def sample_active_subnet_within_range(self, targeted_min_flops, targeted_max_flops):
        while True:
            cfg = self._sample_active_subnet()
            cfg['flops'] = self.compute_active_subnet_flops()
            if cfg['flops'] >= targeted_min_flops and cfg['flops'] <= targeted_max_flops:
                return cfg

    def _sample_active_subnet(self, min_net=False, max_net=False, uniform=False, random_uniform=False, factor=None):
        sample_cfg = lambda candidates, sample_min, sample_max: \
            min(candidates) if sample_min else (max(candidates) if sample_max else random.choice(candidates))
        cfg = {}
        if not (uniform or random_uniform):
            for k in ['g_a', 'h_a', 'g_s', 'h_mean_s', 'h_scale_s']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):
                    cfg[k].append(sample_cfg(self.cfg_candidates[k]["c"][layer_index], min_net, max_net))
        else:
            if random_uniform:
                factor = random.choice(self.uniform_candidates)
            for k in ['g_a', 'h_a', 'g_s', 'h_mean_s', 'h_scale_s']:
                cfg[k] = []
                for layer_index in range(0, self.cfg_candidates[k]['layer_num']):  # the last layer don't partition!
                    if layer_index == self.cfg_candidates[k]['layer_num'] - 1:
                        cfg[k].append(self.cfg_candidates[k]["c"][layer_index][-1])
                    else:
                        cfg[k].append(int(self.cfg_candidates[k]["c"][layer_index][-1] * factor))

        self.set_active_subnet(cfg)
        return cfg

    def compute_active_subnet_flops(self, image_size, g_s=True, h_s=False, bias=False, context_model=False, g_a=False):

        def count_conv(c_in, c_out, size_in, k, groups=1, bias=False, stride=2):
            if isinstance(k, tuple):
                kernel_ops = k[0] * k[1]
            else:
                kernel_ops = k ** 2
            if isinstance(stride, tuple):
                output_elements = c_out * size_in[0] * size_in[1] / stride[0] / stride[1]
            else:
                output_elements = c_out * size_in[0] * size_in[1] / stride / stride
            ops = c_in * output_elements * kernel_ops
            if bias:
                ops += output_elements
            return ops

        def count_tconv(c_in, c_out, size_in, groups, k, bias=False, stride=2):
            if isinstance(k, tuple):
                kernel_ops = k[0] * k[1]
            else:
                kernel_ops = k ** 2
            intput_elements = 0
            if isinstance(stride, tuple):
                intput_elements = c_in * size_in[0] * size_in[1] * stride[0] * stride[1]
            else:
                intput_elements = c_in * size_in[0] * size_in[1] * stride * stride

            ops = intput_elements * kernel_ops * c_out
            if bias:
                if isinstance(stride, tuple):
                    ops += c_out * size_in[0] * size_in[1] * stride[0] * stride[1]
                else:
                    ops += c_out * size_in[0] * size_in[1] * stride * stride
            return ops

        def BottleneckDenseBlock(c_in, c_out, size_in, bias=False, isFusion=True):
            ops = 0
            ops += count_conv(c_in, c_out, size_in, k=1, stride=1, bias=bias) * 2
            ops += count_conv(c_out, c_out, size_in, k=3, stride=1, bias=bias)
            if isFusion:
                ops += count_conv(c_in, c_out, size_in, k=1, stride=1, bias=bias) * 3
            return ops

        total_ops = 0
        if context_model:
            size_in = (image_size[0] // 16, image_size[1] // 16)
            c_in = self.M
            slice_depth = self.M // self.num_slices
            second = count_conv(c_in=224, c_out=128, size_in=size_in, groups=1, k=(3, 3), stride=1, bias=bias)
            third = count_conv(c_in=128, c_out=slice_depth, size_in=size_in, groups=1, k=(3, 3), stride=1, bias=bias)
            for i in range(self.num_slices):
                first = count_conv(c_in=c_in + slice_depth * min(i, self.max_support_slices), c_out=224, size_in=size_in, groups=1, k=(3, 3), stride=1, bias=bias)
                total_ops += first * 2
                first = count_conv(c_in=c_in + slice_depth * min(i + 1, self.max_support_slices + 1), c_out=224, size_in=size_in, groups=1, k=(3, 3), stride=1, bias=bias)
                total_ops += first
                total_ops += second * 3
                total_ops += third * 3

        if h_s:
            size_in = (image_size[0] // 64, image_size[1] // 64)
            c_in = self.N
            for layer_index, layer in enumerate(self.h_scale_s.h_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                 k=config['kernel_size'],
                                                 stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                k=config['kernel_size'],
                                                stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_ops += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_ops += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                             stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                            stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        if h_s:
            size_in = (image_size[0] // 64, image_size[1] // 64)
            c_in = self.N
            for layer_index, layer in enumerate(self.h_mean_s.h_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                 k=config['kernel_size'],
                                                 stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                k=config['kernel_size'],
                                                stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_ops += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_ops += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                             stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                            stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        if g_s:
            size_in = (image_size[0] // 16, image_size[1] // 16)
            c_in = self.M
            for layer_index, layer in enumerate(self.g_s.g_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                 k=config['kernel_size'],
                                                 stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                k=config['kernel_size'],
                                                stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_ops += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_ops += count_conv(c_in, c_out, size_in, k=1, stride=1, bias=bias) * 3
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                             stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                            stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        if g_a:
            size_in = (image_size[0], image_size[1])
            c_in = 3
            for layer_index, layer in enumerate(self.g_a.g_a):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                 k=config['kernel_size'],
                                                 stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                k=config['kernel_size'],
                                                stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_ops += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_ops += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                             stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                            stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
            for layer_index, layer in enumerate(self.h_a.h_a):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                 k=config['kernel_size'],
                                                 stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                k=config['kernel_size'],
                                                stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_ops += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_ops += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_ops += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                             stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_ops += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1, k=config['kernel_size'],
                                            stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        return total_ops

    def compute_active_subnet_parameters(self, image_size, g_s=True, h_s=False, context_model=False, bias=False):

        def count_conv(c_in, c_out, size_in, k, groups=1, bias=False, stride=2):
            if isinstance(k, tuple):
                kernel_ops = k[0] * k[1]
            else:
                kernel_ops = k ** 2
            Parameters = c_in * kernel_ops * c_out
            if bias:
                Parameters += c_out
            return Parameters

        def count_tconv(c_in, c_out, size_in, groups, k, bias=False, stride=2):
            if isinstance(k, tuple):
                kernel_ops = k[0] * k[1]
            else:
                kernel_ops = k ** 2
            Parameters = c_in * kernel_ops * c_out
            if bias:
                Parameters += c_out
            return Parameters

        def BottleneckDenseBlock(c_in, c_out, size_in, bias=False):
            Parameters = 0
            Parameters += count_conv(c_in, c_out, size_in, k=1, stride=1, bias=bias) * 2
            Parameters += count_conv(c_out, c_out, size_in, k=3, stride=1, bias=bias)
            Parameters += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
            return Parameters
        total_parameters = 0
        if context_model:
            total_parameters += sum(p.numel() for p in self.cc_mean_transforms.parameters())
            total_parameters += sum(p.numel() for p in self.cc_scale_transforms.parameters())
            total_parameters += sum(p.numel() for p in self.lrp_transforms.parameters())

        if h_s:
            size_in = (image_size[0] // 64, image_size[1] // 64)
            c_in = self.N
            for layer_index, layer in enumerate(self.h_scale_s.h_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                        k=config['kernel_size'],
                                                        stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                       k=config['kernel_size'],
                                                       stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_parameters += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_parameters += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                    k=config['kernel_size'],
                                                    stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                   k=config['kernel_size'],
                                                   stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        if h_s:
            size_in = (image_size[0] // 64, image_size[1] // 64)
            c_in = self.N
            for layer_index, layer in enumerate(self.h_mean_s.h_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                        k=config['kernel_size'],
                                                        stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                       k=config['kernel_size'],
                                                       stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_parameters += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_parameters += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                    k=config['kernel_size'],
                                                    stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                   k=config['kernel_size'],
                                                   stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        if g_s:
            size_in = (image_size[0] // 16, image_size[1] // 16)
            c_in = self.M
            for layer_index, layer in enumerate(self.g_s.g_s):
                config = layer.config
                c_out = layer.conv.active_out_channel
                if isinstance(layer, IBasicBottleneckDenseBlock):
                    if config['tconv']:
                        total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                        k=config['kernel_size'],
                                                        stride=config['stride'], bias=bias)
                        c_in = c_out
                        size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                    else:
                        total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                       k=config['kernel_size'],
                                                       stride=config['stride'], bias=bias)
                        size_in = (size_in[0] // config['stride'][0], size_in[1] // config['stride'][1])
                        c_in = c_out
                    total_parameters += BottleneckDenseBlock(c_in=c_in, c_out=c_out, size_in=size_in, bias=bias) * 3
                    total_parameters += count_conv(c_in * 3, c_out, size_in, k=1, stride=1, bias=bias)
                elif isinstance(layer, IBasicTransposeConv2D):
                    total_parameters += count_tconv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                    k=config['kernel_size'],
                                                    stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])
                elif isinstance(layer, IBasicConv2D):
                    total_parameters += count_conv(c_in=c_in, c_out=c_out, size_in=size_in, groups=1,
                                                   k=config['kernel_size'],
                                                   stride=config['stride'], bias=bias)
                    c_in = c_out
                    size_in = (size_in[0] * config['stride'][0], size_in[1] * config['stride'][1])

        return total_parameters


if __name__ == "__main__":
    pass
