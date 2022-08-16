import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_gaussian_kernel2d(kernel_size, sigma, normalize=True):
    kernel_1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
    kernel_2d = kernel_1d * kernel_1d.T
    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d


def get_customized_kernel2d(kernel_size, valid_ratio, normalize=False):
    assert 0 <= valid_ratio <= 1
    valid_ratio = -np.log(1 - valid_ratio)
    valid_width = kernel_size // 2 * valid_ratio

    kernel_1d = np.interp(
        x=np.arange(-(kernel_size // 2), 1), xp=[-valid_width, 0], fp=[0, 1]
    )
    kernel_1d = np.concatenate([kernel_1d, kernel_1d[::-1][1:]], axis=-1)
    kernel_1d = kernel_1d.reshape((1, -1))
    kernel_2d = kernel_1d * kernel_1d.T

    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d

def get_customized_kernel2d_pt(kernel_size, valid_ratio, normalize=False):
    assert valid_ratio.min() >= 0 and valid_ratio.max() <= 1
    ori_shape = valid_ratio.shape
    valid_ratio = -torch.log(1 - valid_ratio)

    valid_width = kernel_size // 2 * valid_ratio
    coords = torch.arange(kernel_size).reshape(-1, 1, 1)
    kernel_1d = F.relu(1 - torch.abs(coords / valid_width))
    kernel_2d = torch.einsum("xhw,yhw->xyhw", kernel_1d, kernel_1d)
    kernel_2d = kernel_2d.reshape(-1, *ori_shape)

    if normalize:
        kernel_2d = kernel_2d / kernel_2d.sum(dim=0)
    return kernel_2d


image = "image.jpg"
depth = "depth.png"

image = cv2.imread(image, cv2.IMREAD_COLOR)
depth = cv2.imread(depth, cv2.IMREAD_GRAYSCALE)

# ensure the same shape
height, width, num_channels = image.shape
depth = cv2.resize(depth, dsize=(width, height))

image = torch.from_numpy(image).float()
depth = torch.from_numpy(depth).float()

# normalize the depth image
depth = depth - depth.min()
depth = depth / depth.max()

kernel_size = 21
tgt_image = torch.zeros_like(image, dtype=torch.float32)

unfold_ops = torch.nn.Unfold(kernel_size=kernel_size)

image = image.permute(2, 0, 1).unsqueeze(0)
image = F.pad(image, pad=(kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='replicate')
image = unfold_ops(image)
image = image.reshape(num_channels, kernel_size*kernel_size, height, width)

weight = get_customized_kernel2d_pt(kernel_size=kernel_size, valid_ratio=depth, normalize=True)

tgt_image = (image * weight).sum(dim=1)

tgt_image = tgt_image.permute(1, 2, 0).numpy()
cv2.imwrite("merged_image.png", tgt_image.astype(np.uint8))
