# -*- coding: utf-8 -*-
# @Time    : 2021/7/17
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import itertools

import torch
from torch.utils import data
from torch.utils.data import Sampler


class OurDataset(data.Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return self.tensor[index]

    def __len__(self):
        return len(self.tensor)


class RandomGroupSampler(Sampler[int]):
    r"""Samples elements randomly and maintain the neighbor relationship within the batch.

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): batch size
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, batch_size, generator=None) -> None:
        self.data_source = data_source
        self.batch_size = batch_size
        self.generator = generator

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        return len(self.data_source)

    def __iter__(self):
        # 1. grouping according to the setting batch size
        total_indices = torch.arange(len(self.data_source))
        original_index_groups = torch.split(total_indices, split_size_or_sections=self.batch_size, dim=0)

        # 2. shuffle group indices
        num_groups = len(original_index_groups)
        permuted_group_indices = torch.randperm(num_groups, generator=self.generator).tolist()

        # 3. shuffle groups and chain all group into a 1-D iterable
        shuffled_index_groups = itertools.chain(*[original_index_groups[i].tolist() for i in permuted_group_indices])
        yield from shuffled_index_groups

    def __len__(self):
        return self.num_samples


a = torch.arange(20)
dataset = OurDataset(a)

loader = data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    sampler=RandomGroupSampler(dataset, batch_size=2, generator=None),
)

for epoch in range(3):
    print(f"Epoch: {epoch}")
    for batch in loader:
        print(batch)
