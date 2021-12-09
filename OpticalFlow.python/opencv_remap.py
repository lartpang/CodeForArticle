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

"""
remap(src, map1, map2, interpolation[, dst[, borderMode[, borderValue]]]) -> dst
dst(x,y) =  src(map_x(x,y), map_y(x,y))

array 坐标索引形式: [y, x]
"""
base_coord = np.stack(np.meshgrid(range(10), range(10))[::-1], axis=0)
flow = np.zeros((2, 10, 10), dtype=np.float32)

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
