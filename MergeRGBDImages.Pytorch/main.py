import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_kernel2d(kernel_size, valid_ratio, normalize=False):
    """为局部区域的加权求和生成2d的核权重。

    Args:
        kernel_size (int): 核的边长。
        valid_ratio (torch.Tensor): 有效区间比例（基于边长的一半）。
        normalize (bool, optional): 是否归一化输出。

    Returns:
        torch.Tensor: 生成的核 (kernel_size*kernel_size, H, W)
    """
    assert valid_ratio.min() >= 0 and valid_ratio.max() <= 1
    ori_shape = valid_ratio.shape
    valid_ratio = -torch.log(1 - valid_ratio)

    valid_width = kernel_size // 2 * valid_ratio
    coords = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size).reshape(-1, 1, 1)

    kernel_1d = F.relu(1 - torch.abs(coords / valid_width)).nan_to_num(nan=1)
    kernel_2d = torch.einsum("xhw,yhw->xyhw", kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.reshape(-1, *ori_shape)

    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum(dim=0)
    return kernel_2d

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('rgb', type=str, help='rgb图像的路径')
    parser.add_argument('depth', type=str, help="depth图像的路径")
    parser.add_argument('result', type=str, help="虚化图像的保存路径")
    parser.add_argument('--k', type=int, default=21, help="虚化使用的正方形核边长")
    return parser.parse_args()

def main():
    args = get_args()

    image = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    depth = cv2.imread(args.depth, cv2.IMREAD_GRAYSCALE)
    height, width, num_channels = image.shape

    depth = cv2.resize(depth, dsize=(width, height))
    image = torch.from_numpy(image).float()
    depth = torch.from_numpy(depth).float()
    depth = depth - depth.min()
    depth = depth / depth.max()

    tgt_image = torch.zeros_like(image, dtype=torch.float32)

    image = image.permute(2, 0, 1).unsqueeze(0)
    image = F.pad(image, pad=(args.k//2, args.k//2, args.k//2, args.k//2), mode='replicate')
    image = F.unfold(image, kernel_size=args.k)
    image = image.reshape(num_channels, args.k*args.k, height, width)

    weight = get_kernel2d(kernel_size=args.k, valid_ratio=depth, normalize=True)

    tgt_image = (image * weight).sum(dim=1)
    tgt_image = tgt_image.permute(1, 2, 0).numpy()

    cv2.imwrite(args.result, tgt_image.astype(np.uint8))

if __name__ == '__main__':
    main()
