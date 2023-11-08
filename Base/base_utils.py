import os
import numpy as np
import random
import datetime
import math

import struct
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
import yaml


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def file_path_init(args):
    from glob import glob
    from shutil import copyfile
    date = str(datetime.datetime.now())
    date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    makedirs('./logs')
    args.log_dir = os.path.join(args.log_root, f"{args.CompressorName}_PSNR_{date}")
    makedirs(args.log_dir)
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.data', '.chekpoint', 'logs', 'results', '.gitignore', '.nsmlignore', 'resrc']
    makedirs(os.path.join(args.log_dir, 'codes'))
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        makedirs(os.path.join(args.log_dir, 'codes', to_make))

    # if not args.load_pretrained:
    pyfiles = glob("./*.py")
    for py in pyfiles:
        copyfile(py, os.path.join(args.log_dir, 'codes') + "/" + py)

    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        tmp_files = glob(os.path.join('./', to_make, "*.py"))
        for py in tmp_files:
            copyfile(py, os.path.join(args.log_dir, 'codes', py[2:]))



def get_net_device(net):
    return net.parameters().__next__().device


def make_divisible(v, divisor=8, min_value=1):
    """
    forked from slim:
    https://github.com/tensorflow/models/blob/\
    0344c5503ee55e24f0de7f37336a6e08f10976fd/\
    research/slim/nets/mobilenet/mobilenet.py#L62-L69
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def read_image(filepath):
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def cal_psnr(a, b, select=True):
    if select:
        mse = F.mse_loss(a, b).item()
        return -10 * math.log10(mse)
    else:
        mse = (torch.abs(a - b) * torch.abs(a - b)).mean((1, 2, 3))
        return -10 * torch.log10(mse)



def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def fix_random_seed(seed_value=2021):
    os.environ['PYTHONPATHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)

    # torch
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return 0


def save_weights(name, model, optim, scheduler, root, iteration):
    path = os.path.join(root, "{}_weights.pth".format(name))
    state = dict()
    state["name"] = name
    state["iteration"] = iteration
    state["modelname"] = model.__class__.__name__
    state["model"] = model.state_dict()
    state["optim"] = optim.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    else:
        state["scheduler"] = None
    torch.save(state, path)


def save_model(root, name, model):
    path = os.path.join(root, "{}_all.pth".format(name))
    torch.save(model, path)


def load_state(path, cuda):
    if cuda:
        print("INFO: [*] Load Mode To GPU")
        state = torch.load(path)
    else:
        print("INFO: [*] Load To CPU")
        state = torch.load(path, map_location=lambda storage, loc: storage)
    return state


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def rename_key(key):
    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"
    return key


def write_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, path)


def read_yaml(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            return data
    except:
        return None



def load_pretrained(state_dict):
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def Average(lst):
    return sum(lst) / len(lst)


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def show_image(img: Image.Image):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def calculate_enthroy(image, w):
    data = np.array(image * 255).astype("uint8").clip(0, 254)
    count_B = np.zeros(255)
    count_G = np.zeros(255)
    count_R = np.zeros(255)

    for i in range(0, w):
        for j in range(0, w):
            count_R[data[0][0][i][j]] = count_R[data[0][0][i][j]] + 1

    for i in range(0, w):
        for j in range(0, w):
            count_G[data[0][1][i][j]] = count_G[data[0][0][i][j]] + 1

    for i in range(0, w):
        for j in range(0, w):
            count_B[data[0][2][i][j]] = count_B[data[0][0][i][j]] + 1

    for i in range(1, 256):
        count_B[i - 1] = count_B[i - 1] / w / w
        count_G[i - 1] = count_G[i - 1] / w / w
        count_R[i - 1] = count_R[i - 1] /  w / w
    H_B=0
    H_G=0
    H_R=0
    for i in range(1,256):
        if(count_B[i - 1]==0):
            H_B = H_B + 0
        else:
            H_B = H_B + count_B[i - 1] * math.log2(count_B[i - 1])
        if (count_G[i - 1] == 0):
            H_G = H_G + 0
        else:
            H_G = H_G + count_G[i - 1] * math.log2(count_G[i - 1])
        if (count_R[i - 1] == 0):
            H_R = H_R + 0
        else:
            H_R = H_R + count_R[i - 1] * math.log2(count_R[i - 1])
    H_B=-H_B
    H_G=-H_G
    H_R=-H_R
    return (H_R+H_G+H_B) / 3

#用PIL库批量裁剪指定大小的图像(自动填充）
def img_crop(img_path):
    new_img = []
    img = Image.open(img_path).convert("RGB")
    # img = transforms.ToTensor()(img)
    width, hight = img.size
    w = 128  # 需要切成图片块的大小，默认大小为w*w,可以自己设置
    id = 1
    i = 0
    padw = padh = 0  # 当宽高除不尽切块大小时，对最后一块进行填充
    if width % w != 0:
        padw = 1  # 宽除不尽的情况
    if hight % w != 0:
        padh = 1  # 高除不尽的情况

    # 默认从最左上角向右裁剪，再向下裁剪
    while i + w <= hight:
        j = 0
        while j + w <= width:
            new_img.append(transforms.ToTensor()(img.crop((j, i, j + w, i + w))).unsqueeze(0))
            id += 1
            j += w
        if padw == 1:  # 宽有除不尽的情况
            new_img.append(transforms.ToTensor()(img.crop((width - w, i, width, i + w))).unsqueeze(0))
            id += 1
        i = i + w

    if padh == 1:  # 高除不尽的情况
        j = 0
        while j + w <= width:
            new_img.append(transforms.ToTensor()(img.crop((j, hight - w, j + w, hight))).unsqueeze(0))
            id += 1
            j += w
        if padw == 1:
            new_img.append(transforms.ToTensor()(img.crop((width - w, hight - w, width, hight))).unsqueeze(0))
            id += 1
    img_crop = new_img[0]
    for i in range(1, len(new_img)):
        img_crop = torch.concat((img_crop, new_img[i]), dim=0)
    enthroy = []
    for i in range(0, len(new_img)):
        enthroy.append(calculate_enthroy(new_img[i], w))
    img = transforms.ToTensor()(img).unsqueeze(0)
    return img_crop, img, enthroy

def img_restore(image, image_org):
    w = 128
    width, height = image_org.size()[2], image_org.size()[3]
    width_w_num = width // w
    height_w_num = height // w
    width_list = []
    for i in range(width_w_num):
        data = image[i * height_w_num, :, :, :]
        for j in range(1, height_w_num):
            data = torch.concat((data, image[i * height_w_num + j, :, :, :]), dim=2)
        width_list.append(data)

    result = width_list[0]
    for i in range(1, width_w_num):
        result = torch.concat((result, width_list[i]), dim=1)
    return result.unsqueeze(0)