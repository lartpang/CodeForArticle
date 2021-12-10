# -*- coding: utf-8 -*-
# @Time    : 2020/12/13
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
import time
from math import prod

import torch
from torch.nn.functional import one_hot


def bhw_to_onehot_by_for(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:
    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = bhw_tensor.new_zeros(size=(num_classes, *bhw_tensor.shape))
    for i in range(num_classes):
        one_hot[i, bhw_tensor == i] = 1
    one_hot = one_hot.permute(1, 2, 3, 0)
    return one_hot


def bhw_to_onehot_by_for_V1(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:
    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = bhw_tensor.new_zeros(size=(*bhw_tensor.shape, num_classes))
    for i in range(num_classes):
        one_hot[..., i][bhw_tensor == i] = 1
    return one_hot


def bhw_to_onehot_by_scatter(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:
    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = torch.zeros(size=(prod(bhw_tensor.shape), num_classes), device=bhw_tensor.device, dtype=bhw_tensor.dtype)
    one_hot.scatter_(dim=1, index=bhw_tensor.reshape(-1, 1), value=1)
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    return one_hot


def bhw_to_onehot_by_scatter_V1(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:
    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = torch.zeros(size=(*bhw_tensor.shape, num_classes), device=bhw_tensor.device, dtype=bhw_tensor.dtype)
    one_hot.scatter_(dim=-1, index=bhw_tensor[..., None], value=1)
    return one_hot


def bhw_to_onehot_by_index_select(bhw_tensor: torch.Tensor, num_classes: int):
    """
    Args:
        bhw_tensor: b,h,w
        num_classes:
    Returns: b,h,w,num_classes
    """
    assert bhw_tensor.ndim == 3, bhw_tensor.shape
    assert num_classes > bhw_tensor.max(), torch.unique(bhw_tensor)
    one_hot = torch.eye(num_classes, device=bhw_tensor.device, dtype=bhw_tensor.dtype)
    one_hot = one_hot.index_select(dim=0, index=bhw_tensor.reshape(-1))
    one_hot = one_hot.reshape(*bhw_tensor.shape, num_classes)
    return one_hot


def test(a):
    times = []

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    b = bhw_to_onehot_by_for(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("bhw_to_onehot_by_for", time.time() - start))

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    b1 = bhw_to_onehot_by_for_V1(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("bhw_to_onehot_by_for_V1", time.time() - start))

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    c = bhw_to_onehot_by_scatter(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("bhw_to_onehot_by_scatter", time.time() - start))

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    c1 = bhw_to_onehot_by_scatter_V1(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("bhw_to_onehot_by_scatter_V1", time.time() - start))

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    d = bhw_to_onehot_by_index_select(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("bhw_to_onehot_by_index_select", time.time() - start))

    if a.is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    e = one_hot(a, num_classes=num_classes)
    if a.is_cuda:
        torch.cuda.synchronize()
    times.append(("F.one_hot", time.time() - start))

    return b, b1, c, c1, d, e, times


if __name__ == "__main__":
    print(torch.__version__, torch.cuda.get_device_name())

    num_classes = 20
    data = dict(
        cpu=torch.randint(high=num_classes, low=0, size=(4, 1000, 1000), dtype=torch.long),
        gpu=torch.randint(high=num_classes, low=0, size=(4, 1000, 1000), dtype=torch.long).cuda(),
    )
    for dev, a in data.items():
        print(dev)

        # warmup
        for _ in range(5):
            test(a)

        b, b1, c, c1, d, e, times = test(a)
        for t in times:
            print(t)
        print(torch.all(torch.isclose(e, b)))
        print(torch.all(torch.isclose(e, b1)))
        print(torch.all(torch.isclose(e, c)))
        print(torch.all(torch.isclose(e, c1)))
        print(torch.all(torch.isclose(e, d)))
