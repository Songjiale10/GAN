import numpy as np
import torch.utils.data
from torchvision import transforms
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import os

"""数据加载代码"""
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=0.5)  # 将输入图片的取值规范到[-1,1] 设置均值和方差为0.5
    ]
)
train_ds = torchvision.datasets.MNIST(
    'data/',
    train=True,
    transform=transform,
    download=True
)
"""DataLoader接收数据集，并进行处理。此处为小批次处理和打乱，以便于从数据集中获取数据
接收来自用户的Dataset实例，并使用采样器策略将数据采样为小批次"""
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

"""生成器"""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28 * 1),
            nn.Tanh()
        )

    def forward(self, x):
        img = self.main(x)
        img = img.view(-1, 28, 28, 1)  # 改变形状为手写图片的大小
        return img


"""判别器"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28*1, 512),
            nn.LeakyReLU(),
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 1)
        x = self.main(x)
        return x


"""定义损失函数、初始化两个模型和两个优化器"""
device = "cuda" if torch.cuda.is_available() else "cpu"
gen = Generator().to(device)  # 实例化生成器
dis = Discriminator().to(device)  # 实例化判别器
loss_fn = torch.nn.BCELoss()  # 定义损失函数
d_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0001)  # 判别模型的优化器
g_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0001)  # 生成模型的优化器


# 定义可视化函数
def generate_and_save_images(model, epoch, test_input):
    predictions = np.squeeze(model(test_input).detach().cpu().numpy()) #阻断反向传播，把图片转移到cpu上，最后把cpu上的tensor转为numpy数据
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] + 1) / 2, cmap='gray')

        plt.axis('off')
        img_save_path="output_img"
        plt.savefig(os.path.join(img_save_path, 'img_at_epoch_{:04d}.png'.format(epoch)))
    plt.show()


# 设置生成绘图图片的随即张良，这里可视化为16张图片
# 生成16个长度为100的随即正态分布张量
test_seed = torch.randn(16, 100, device=device) #返回一个符合均值为0，方差为1的正态分布（标准正态分布）中填充随机数的张量
