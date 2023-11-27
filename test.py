a=[1,2,3,4,5,6]
#print(a[-1])
#a[:3,:]
#print(a[:3,:])
"""
import torch
device="cuda" if torch.cuda.is_available() else "cpu"
print(device)

"""
"""
最简单的序贯模型
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(1, 20, 5),
    nn.ReLU(),
    nn.Conv2d(20, 64, 5),
    nn.ReLU()
)

print(model)
"""
"""
#Sequential可以用来定义模型之外，它还可以包装层，把几个层包装起来像一个块一样
import torch.nn as nn
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('conv1',nn.Conv2d(1, 20, 5)),
    ('relu1',nn.ReLU()),
    ('conv2',nn.Conv2d(20, 64, 5)),
    ('relu2',nn.ReLU())
])

)

print(model)
"""
"""
import torch
torch.set_printoptions(profile="full")
#torch.set_printoptions(profile="default") # reset
test_seed = torch.randn(16, 100)
print(test_seed)
"""
"""
import torch
import torch.nn as nn

# 假设有一个概率值和一个全1张量
# 用随机生成的数据进行示例
probabilities = torch.tensor([0.8, 0.6, 0.9])  # 预测的概率值
labels = torch.ones_like(probabilities)  # 全1张量作为真实标签

# 创建交叉熵损失函数
loss_func = nn.BCELoss()

# 计算交叉熵损失
loss = loss_func(probabilities, labels)

print(loss.item())  # 打印损失值

import torch
import torch.nn as nn

# 假设有一个概率值和一个全0张量
# 用随机生成的数据进行示例
probabilities = torch.tensor([0.3, 0.2, 0.6])  # 生成器输出的概率值
labels = torch.zeros_like(probabilities)  # 全0张量作为标签

# 创建交叉熵损失函数
loss_func = nn.BCELoss()

# 计算交叉熵损失
loss = loss_func(probabilities, labels)

print(loss.item())  # 打印损失值

"""
import numpy as np
import gzip
"""
# 定义函数加载 MNIST 数据集
def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data
"""

"""
# 指定 MNIST 数据集文件路径
train_images_file = 'train-images-idx3-ubyte.gz'
train_labels_file = 'train-labels-idx1-ubyte.gz'
test_images_file = 't10k-images-idx3-ubyte.gz'
test_labels_file = 't10k-labels-idx1-ubyte.gz'

# 加载训练集图像和标签数据
train_images = load_mnist_images(train_images_file)
train_labels = load_mnist_labels(train_labels_file)

# 加载测试集图像和标签数据
test_images = load_mnist_images(test_images_file)
test_labels = load_mnist_labels(test_labels_file)

# 查看数据集的形状和内容
print("训练集图像形状:", train_images.shape)
print("训练集标签形状:", train_labels.shape)
print("测试集图像形状:", test_images.shape)
print("测试集标签形状:", test_labels.shape)

# 打印第一张图像和对应的标签
print("第一张训练集图像:")
print(train_images[0])
print("第一张训练集标签:", train_labels[0])
"""
import torch
print(torch.randn([10,8]))

