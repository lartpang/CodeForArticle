# Random Group Sampler

A `Random Group Sampler` for sampling elements randomly and maintaining the neighbor relationship within the batch.

Here is a simple demo.

```text
Epoch: 0
tensor([12, 13])
tensor([14, 15])
tensor([6, 7])
tensor([4, 5])
tensor([0, 1])
tensor([8, 9])
tensor([10, 11])
tensor([2, 3])
tensor([16, 17])
tensor([18, 19])
Epoch: 1
tensor([10, 11])
tensor([14, 15])
tensor([18, 19])
tensor([8, 9])
tensor([0, 1])
tensor([12, 13])
tensor([4, 5])
tensor([2, 3])
tensor([6, 7])
tensor([16, 17])
Epoch: 2
tensor([8, 9])
tensor([0, 1])
tensor([16, 17])
tensor([10, 11])
tensor([12, 13])
tensor([4, 5])
tensor([6, 7])
tensor([2, 3])
tensor([14, 15])
tensor([18, 19])
```
