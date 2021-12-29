import torch
from torch.utils.checkpoint import checkpoint

initial_usage = torch.cuda.memory_allocated()
print("0", initial_usage)  # 0

# 模型初始化
linear1 = torch.nn.Linear(1024, 1024, bias=False).cuda()
after_init_linear1 = torch.cuda.memory_allocated()
print("1", after_init_linear1 - initial_usage, linear1.weight.numel())  # 4194304 1048576

linear2 = torch.nn.Sequential(torch.nn.Linear(1024, 1024, bias=False), torch.nn.Linear(1024, 1, bias=False)).cuda()
after_init_linear2 = torch.cuda.memory_allocated()
print("2", after_init_linear2 - after_init_linear1, sum([m.weight.numel() for m in linear2]))  # 4198400 1049600

# 输入定义
inputs = torch.randn(size=(1024, 1024), device="cuda:0")
after_init_inputs = torch.cuda.memory_allocated()
print("3", after_init_inputs - after_init_linear2, inputs.numel())  # 4194304 1048576

print("Iter: 0")

# 前向传播
o = linear1(inputs)
after_linear1 = torch.cuda.memory_allocated()
print("4", after_linear1 - after_init_inputs, o.numel())  # 4194304 1048576

o = checkpoint(linear2, o)
after_linear2 = torch.cuda.memory_allocated()
# 4096 1024 这里使用了checkpoint，可以看到这里并没有存储linear2内部的结果，仅包含输出o
print("5", after_linear2 - after_linear1, o.numel())

"""
在PyTorch中，显存是按页为单位进行分配的，这可能是CUDA设备的限制。
就算我们只想申请4字节的显存，pytorch也会先向CUDA设备申请2MB的显存到自己的cache区中，
然后pytorch再为我们分配512字节或者1024字节的空间。
这个在使用torch.cuda.memory_allocated()的时候可以看出来512字节；
用torch.cuda.memory_cached()可以看出向CUDA申请的2MB。
"""
loss = sum(o)
after_loss = torch.cuda.memory_allocated()
# 16785920 512
print("6", after_loss, after_loss - after_linear2)

# 后向传播
"""
后向传播会将模型的中间激活值给消耗并释放掉掉，并为每一个模型中的参数计算其对应的梯度。
在第一次执行的时候，会为模型参数（即叶子结点）分配对应的用来存储梯度的空间。
所以第一次之后，仅有中间激活值空间在变换。
"""
loss.backward()
after_backward = torch.cuda.memory_allocated()
# 20984320 4198400=-4194304(释放linear1输出的o)+4194304(申请linear1权重对应的梯度)+4198400(申请linear2权重对应的梯度)
# 由于checkpoint的使用，所以linear2没有存储中间激活值，但是保留了最终的激活值，因为变量o对其引用依然在，所以linear2的输出未被释放。
# linear1本身不涉及到中间激活值，而其输出则由于变量o指向了新的内存，所以会被自动回收。
print("7", after_backward, after_backward - after_loss)

print("Iter: 1")

# 前向传播
o = linear1(inputs)
after_linear1 = torch.cuda.memory_allocated()
print("8", after_linear1 - after_backward, o.numel())  # 4190208 1048576

o = checkpoint(linear2, o)
after_linear2 = torch.cuda.memory_allocated()
# 4096 1024
print("9", after_linear2 - after_linear1, o.numel())

"""
因为前一次计算的loss的引用还在，所以这里没有再新申请空间。
"""
loss = sum(o)
after_loss = torch.cuda.memory_allocated()
print("10", after_loss, after_loss - after_linear2)  # 25178624 0

# 后向传播
loss.backward()
after_backward = torch.cuda.memory_allocated()
# 20984320 -4194304
# 这减去部分的恰好等于中间激活值的占用：-4190208(linear1的输出o)-4096(linear2输出o)
# 这里的linaer2使用了checkpoint，则不存linear2中间特征的额外占用，因为这部分是在运算内部申请并实时释放的
print("11", after_backward, after_backward - after_loss)

del loss

print("Iter: 2")

# 前向传播
o = linear1(inputs)
after_linear1 = torch.cuda.memory_allocated()
print("12", after_linear1 - after_backward, o.numel())  # 4190208 1048576

o = linear2(o)
after_linear2 = torch.cuda.memory_allocated()
# 4198400=1024*1024*4(linear2的中间特征)+1024*4(linear2输出o) 1024
print("13", after_linear2 - after_linear1, o.numel())

"""
在前一次计算后，del loss的话，可以看到这里会申请512字节的空间
"""
loss = sum(o)
after_loss = torch.cuda.memory_allocated()
print("14", after_loss, after_loss - after_linear2)  # 29372928 512

# 后向传播
loss.backward()
after_backward = torch.cuda.memory_allocated()
# 20984320 -8388608
# 这减去部分的恰好等于中间激活值的占用：-4190208(linear1的输出o)-4194304(1024*1024*4(linear2中间特征))-4096(linear2输出o)
print("15", after_backward, after_backward - after_loss)
