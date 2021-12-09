# 关于光流的一些记录

## 这样的光流可视化图里的颜色表示啥意思？

> 这样的光流可视化图里的颜色表示啥意思？ - 人民艺术家的回答 - 知乎 https://www.zhihu.com/question/500159852/answer/2233549991

可以参考mmcv的可视化光流的代码：
<https://github.com/open-mmlab/mmcv/blob/c47c9196d067a0900b7b8987a8e82768edab2fff/mmcv/visualization/optflow.py#L24-L73>

```python
def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image.

    Args:
        flow (ndarray): Array of optical flow.
        color_wheel (ndarray or None): Color wheel used to map flow field to
            RGB colorspace. Default color wheel will be used if not specified.
        unknown_thr (str): Values above this threshold will be marked as
            unknown and thus ignored.

    Returns:
        ndarray: RGB image that can be visualized.
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (
        np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
        (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)  # 使用最大模长来放缩坐标值
        dx /= max_rad
        dy /= max_rad

    rad = np.sqrt(dx**2 + dy**2)  # HxW
    angle = np.arctan2(-dy, -dx) / np.pi  # HxW（-1, 1]

    bin_real = (angle + 1) / 2 * (num_bins - 1)  # HxW (0, num_bins-1]
    bin_left = np.floor(bin_real).astype(int)  # HxW 0,1,...,num_bins-1
    bin_right = (bin_left + 1) % num_bins  # HxW 1,2,...,num_bins % num_bins -> 1, 2, ..., num_bins, 0
    w = (bin_real - bin_left.astype(np.float32))[..., None]  # HxWx1
    flow_img = (1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]  # 线性插值计算实际的颜色值
    small_ind = rad <= 1  # 以模长为1作为分界线来分开处理，个人理解这里主要是用来控制颜色的饱和度，而前面的处理更像是控制色调。
    # 小于1的部分拉大
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    # 大于1的部分缩小
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img
```
    
我根据我的直观的理解，补充了一些注释，当然理解有误的地方欢迎指出，我也不是做光流的，可能有些细节理解有误。

可以看到，主要是基于得到的flow数组进行了偏移向量的模长和角度的计算。并根据角度指定颜色，根据模长对颜色强度进行调整。

对应于转换得到的彩色图像，我们可以看到的最直观的就是鲜艳程度，这实际上可以认为对应着偏移的欧氏距离的相对大小（向量模长）。颜色越浅，偏移量越小。

但是偏移方向并不直观。

在光流可视化RGB图中另一点值得关注的移动目标的边缘的清晰程度。一般而言，好的光流算法，边缘会很清晰，可以看看RAFT的结果，相较于PWC-Net要清晰完美的多。

## 两张图像的光流，I0到I1的光流和I1到I0的光流是什么关系？

> 两张图像的光流，I0到I1的光流和I1到I0的光流是什么关系？ - 人民艺术家的回答 - 知乎 https://www.zhihu.com/question/504893403/answer/2263988696

我们可以简单推理下：

首先给定对应关系：图像I1上点(h0,w0)对应于图像I2上点(h1,w1)。 且基于相同的坐标系（图像左上角为(0,0)，左下角为(H,0)）。
- I0到I1的光流可以看做是给定(h0,w0)得到(h1,w1)的运算：h1=h0+h', w1=w0+w'。
- I1到I0的光流可以看做是给定(h1,w1)得到(h0,w0)的运算：h0=h1-h', w0=w1-w'。

虽然不同的库可能对于光流数组的形式要求不同，但是计算模式基本类似的。由于opencv的remap函数要求的是直接指定前后位置，所以可能与这里的推导不同。但是实际上前后坐标的对应关系是反映了这一点的。所以数学运算的角度上，光流取个负值即可。

我们简单基于opencv、numpy和matplotlib试验下：

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_array(arrays):
    fig, axes = plt.subplots(nrows=1, ncols=len(arrays))
    axes = axes.flat
    for ax, (name, array) in zip(axes, arrays.items()):
        ax.imshow(array)
        ax.set_xticks(np.linspace(0, 9, 10))
        ax.set_yticks(np.linspace(0, 9, 10))
        ax.set_title(name)
        ax.grid()
    plt.show()


background = np.zeros((10, 10))
I0 = background.copy()
I0[3, 4] = 1
I1 = background.copy()
I1[4, 7] = 1

base_coord = np.stack(np.meshgrid(range(10), range(10))[::-1], axis=0)
flow = np.zeros((2, 10, 10), dtype=np.float32)

"""
remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst
dst(x,y) =  src(map_x(x,y), map_y(x,y))
"""

flow_I0_to_I1 = flow.copy()
flow_I0_to_I1[:, 4, 7] = (-1, -3)
flow_I0_to_I1 += base_coord
print(flow_I0_to_I1)
I0_to_I1 = cv2.remap(I0, map1=flow_I0_to_I1[1], map2=flow_I0_to_I1[0], interpolation=cv2.INTER_NEAREST)

flow_I1_to_I0 = flow.copy()
flow_I1_to_I0[:, 3, 4] = (1, 3)
flow_I1_to_I0 += base_coord
I1_to_I0 = cv2.remap(I1, map1=flow_I1_to_I0[1], map2=flow_I1_to_I0[0], interpolation=cv2.INTER_NEAREST)

show_array({"I0": I0, "I1": I1, "I0_to_I1": I0_to_I1, "I1_to_I0": I1_to_I0})
```

![](https://pica.zhimg.com/80/v2-60ea9367b265535f5b05e04e465fd20a_720w.jpg)

由于这里是只对偏移后的新位置上进行了调整，而其他位置仍然使用的是原始坐标上的值，所以后两个图中同时会显示原始的点。
