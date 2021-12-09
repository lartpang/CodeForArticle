# 关于光流的一些记录

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
