import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import time
import compressai
from Base.base_utils import load_pretrained, pad
from Base.base_utils import ms_ssim, read_image, cal_psnr, torch2img
import numpy as np
import json
import math
from flexible_image_model import *
import torch
torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

def model_test():
    compressai.set_entropy_coder("ans")
    curr_cfg = {'g_a': [192, 192, 192, 320], 'h_a': [],
                'g_s': [int(192), int(192), int(192), 3],
                'h_scale_s': [int(192), int(256), 320],
                'h_mean_s': [int(192), int(256), 320]}
    _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
    img_path = './kodim'
    ckpt = f'./High/checkpoint_latest.pth'
    codec = GainCA(rate_point=5)
    codec.eval()
    state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
    state_dict = load_pretrained(state_dict)
    codec = codec.cuda()
    codec.load_state_dict(state_dict)
    codec.update(force=True)
    interpolation_coefficient = 0
    for quality in range(0, 5):
        psnr_block_quality = []
        bpp_block_quality = []
        string_path = f'./output/bin/{quality}'
        os.makedirs(string_path, exist_ok=True)
        rec_path = f'./output/rec//{quality + interpolation_coefficient}'
        os.makedirs(rec_path, exist_ok=True)
        codec.set_rate_level(quality, interpolation_coefficient=interpolation_coefficient, isInterpolation=False)
        codec.set_active_subnet(curr_cfg)

        images = glob.glob(os.path.join(img_path, '*.*'))
        print(f'* Find {len(images)} Images')
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        image_num = 0
        for path in images:
            name = path.split('/')[-1].split('.')[0]
            image_num += 1
            image_org = read_image(path).unsqueeze(0)
            image_org = image_org.cuda()
            num_pixels = image_org.size(0) * image_org.size(2) * image_org.size(3)
            image_org = pad(image_org)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                out_enc = codec.compress(image_org)
                torch.cuda.synchronize()
                enc_time = time.time() - start
                encT.append(enc_time)

                torch.cuda.synchronize()
                start = time.time()
                out_dec = codec.decompress(out_enc["strings"], out_enc["shape"])
                torch.cuda.synchronize()
                dec_time = time.time() - start
                decT.append(dec_time)
                bpp = 0
                for s in out_enc["strings"]:
                    for i in range(len(s)):
                        bpp += len(s[i])
                bpp = bpp * 8.0 / num_pixels
                psnr = cal_psnr(image_org, out_dec["x_hat"])
                bpp_block_quality.append(bpp)
                psnr_block_quality.append(psnr)
                # x_hat = torch2img(out_dec["x_hat"])
                # x_hat.save(rec_path + '/' + name + '.png')
                ms_ssim1 = ms_ssim(image_org, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)

            print(
                f"{name} | Quality {quality + interpolation_coefficient} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                f"| MS-SSIM {-10 * math.log10(1 - ms_ssim1):.4f} | PSNR {psnr:.4f} | MS-SSIM {ms_ssim1:.5f}"
            )
        print(
            f'Quality {quality + interpolation_coefficient} | Average BPP {np.mean(Bpp):.4f} | PSNR {np.mean(PSNR):.4f} | MSSSIM {-10 * math.log10(1 - np.mean(MS_SSIM)):.4f}'
            f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')
        _PSNR.append(np.mean(PSNR))
        _MS_SSIM.append(-10 * math.log10(1 - np.mean(MS_SSIM)))
        _Bpp.append(np.mean(Bpp))
        _encT.append(np.mean(encT))
        _decT.append(np.mean(decT))

    print(f'BPP: {_Bpp}')
    print(f'PSNR : {_PSNR}')
    print(f'MSSSIM : {_MS_SSIM}')


def model_test_IDCA():
    compressai.set_entropy_coder("ans")
    _Bpp, _MS_SSIM, _PSNR, _encT, _decT = [], [], [], [], []
    img_path = './kodim'
    ckpt = f'./High/checkpoint_latest.pth'
    codec = GainCA(rate_point=5)
    codec.eval()
    state_dict = torch.load(ckpt, map_location='cpu')["state_dict"]
    state_dict = load_pretrained(state_dict)
    codec = codec.cuda()
    codec.load_state_dict(state_dict)
    codec.update(force=True)

    lambdaC = 0.5
    with open(os.path.join('./', f'search_result.json'), 'r') as f:
        search_result = json.load(f)
        print(len(search_result))
    codec.sample_active_subnet(mode='largest')  # the largest complexity

    interpolation_coefficient = 0
    for quality in range(0, 5):
        psnr_block_quality = []
        bpp_block_quality = []
        string_path = f'./output/bin/{quality}'
        os.makedirs(string_path, exist_ok=True)
        rec_path = f'./output/rec/Kodak_Complexity/{quality + interpolation_coefficient}/{71}'
        os.makedirs(rec_path, exist_ok=True)

        codec.set_rate_level(quality, interpolation_coefficient=interpolation_coefficient, isInterpolation=False)
        images = glob.glob(os.path.join(img_path, '*.*'))
        print(f'* Find {len(images)} Images')
        Bpp, MS_SSIM, PSNR, encT, decT = [], [], [], [], []
        image_num = 0
        for path in images:
            name = path.split('/')[-1].split('.')[0]
            image_num += 1
            image_org = read_image(path).unsqueeze(0)
            image_org = image_org.cuda()
            num_pixels = image_org.size(0) * image_org.size(2) * image_org.size(3)
            image_org = pad(image_org)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                out_enc = codec.IDCA_compress(image_org, search_result, quality, 0)
                torch.cuda.synchronize()
                enc_time = time.time() - start
                encT.append(enc_time)

                torch.cuda.synchronize()
                start = time.time()
                out_dec = codec.IDCA_decompress(out_enc["strings"], out_enc["shape"],
                                                quality, 0, all_cfg=search_result, quality_list=out_enc['quality_list'], lamdaC=lambdaC)
                torch.cuda.synchronize()
                dec_time = time.time() - start
                decT.append(dec_time)
                bpp = 0
                for s in out_enc["strings"]:
                    for i in range(len(s)):
                        bpp += len(s[i])
                bpp = bpp * 8.0 / num_pixels
                psnr = cal_psnr(image_org, out_dec["x_hat"])
                bpp_block_quality.append(bpp)
                psnr_block_quality.append(psnr)
                ms_ssim1 = ms_ssim(image_org, out_dec["x_hat"], data_range=1.0).item()
                PSNR.append(psnr)
                Bpp.append(bpp)
                MS_SSIM.append(ms_ssim1)

            print(
                f"{name} | Quality {quality + interpolation_coefficient} | {bpp:.4f} bpp | Encoded in {enc_time:.3f}s | Decoded in {dec_time:.3f}s "
                f"| MS-SSIM {-10 * math.log10(1 - ms_ssim1):.4f} | PSNR {psnr:.4f} | MS-SSIM {ms_ssim1:.5f}"
            )
        print(
            f'Quality {quality + interpolation_coefficient} | Average BPP {np.mean(Bpp):.4f} | PSNR {np.mean(PSNR):.4f} | MSSSIM {-10 * math.log10(1 - np.mean(MS_SSIM)):.4f}'
            f' | Encode Time {np.mean(encT):.4f} | Decode Time {np.mean(decT):.4f}')
        _PSNR.append(np.mean(PSNR))
        _MS_SSIM.append(-10 * math.log10(1 - np.mean(MS_SSIM)))
        _Bpp.append(np.mean(Bpp))
        _encT.append(np.mean(encT))
        _decT.append(np.mean(decT))

    print(f'BPP: {_Bpp}')
    print(f'PSNR : {_PSNR}')
    print(f'MSSSIM : {_MS_SSIM}')


if __name__ == "__main__":
    model_test_IDCA()
