import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

input_dim = 100
batch_size = 128
num_epoch = 20


# =================================================生成器================================================================
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(  # br1和br2是进行归一化操作
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.br1(self.fc1(x))
        x = self.br2(self.fc2(x))
        x = x.reshape(-1, 128, 7, 7)
        x = self.conv1(x)  # 通过转置卷积层conve1进行反卷积操作
        output = self.conv2(x) #输出通道数为1,所以是灰度图像
       # print(output)
        return output


# =================================================判别器================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.pl1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1),
            nn.LeakyReLU(0.2)
        )
        self.pl2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 4 * 4, 1024),
            nn.LeakyReLU(0.2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.pl1(x)
        x = self.conv2(x)
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        output = self.fc2(x)
        return output  #最后输出一个标量值，表示输入数据被判别为真实图像的概率。


# ==================================================训练================================================================
def training(x):  #返回G和D的loss,调用时传入的就是真实数据
    '''判别器'''
    real_x = x.to(device)
    real_output = D(real_x)
    real_loss = loss_func(real_output, torch.ones_like(real_output).to(device))
    """detach返回一个新的tensor，新的tensor和原来的tensor共享数据内存，但不涉及梯度计算"""
    """生成了一个大小为[batch_size, input_dim]的随机噪声张量，其中batch_size表示批大小，input_dim表示噪声向量的维度。"""
    fake_x = G(
        #几个参数几维
        torch.randn([batch_size, input_dim]).to(device)).detach()  # size是两个数，理解为batch_size行，input_dim列的张量。里面的元素服从方差为1的正态分布，本句话是G生成一张加图片，一批次中有batch_size个
    fake_output = D(fake_x)
    #print(fake_output)
    fake_loss = loss_func(fake_output, torch.zeros_like(fake_output).to(device))
    loss_D = real_loss + fake_loss

    optim_D.zero_grad()
    loss_D.backward()#将损失loss向输入测进行反向传播，同时对需要进行梯度计算的所有变量计算梯度，并进行累积
    optim_D.step()#优化器对参数进行更新， 更新的是 optim_G 管理的参数，即生成器 G 的参数

    '''生成器生成新图像'''
    fake_x = G(torch.randn([batch_size, input_dim]).to(device))#传入G的是一个随机取直的张量（方差为1）
    fake_output = D(fake_x)

    loss_G = loss_func(fake_output, torch.ones_like(fake_output).to(device))

    optim_G.zero_grad()#计算生成器损失函数关于参数的梯度，
    loss_G.backward()#将损失loss向输入测进行反向传播，同时对需要进行梯度计算的所有变量计算梯度，并进行累积
    optim_G.step()#优化器对参数进行更新

    return loss_D, loss_G

"""
# 独热编码
# 输入x代表默认的torchvision返回的类比值，class_count类别值为10
def one_hot(x, class_count=10):
    #torch.eye输出对角线是1，其余全是0的二维数组
    return torch.eye(class_count)[x, :]  # 切片选取，第一维选取第x个，第二维全要---单位矩阵取任意一行都是独热编码


#数据加载代码
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(0.5, 0.5)])  # transforms.Compose串联多个图片变换的操作，此处为转为Tensor然后归一化

# dataset =
# dataloader = datasets.DataLoader(dataset, batch_size=64, shuffle=True)
"""
if __name__ == '__main__':
    # train_dataset = torchvision.datasets.MNIST('data',
    #                                  train=True,
    #                                  transform=transform,
    #                                  target_transform=one_hot,
    #                                  download=True)
    # train_loader = datasets.DataLoader(train_dataset, batch_size=64, shuffle=True)
    print(datasets.MNIST)
    train_dataset = datasets.MNIST(root="./data/", train=True, transform=transforms.ToTensor(), download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True) #每个批次都是一个包含128个样本的小批量

    G = Generator(input_dim).to(device)  # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
    D = Discriminator().to(device)
    """通过使用优化器 optim_G，可以在训练过程中使用 optim_G.zero_grad() 清零梯度，
    然后使用 loss_G.backward() 计算生成器损失函数关于参数的梯度，
    最后使用 optim_G.step() 更新生成器的参数，从而实现生成器的参数优化。(training中)"""
    optim_G = torch.optim.Adam(G.parameters(), lr=0.0002)#lr是学习率
    optim_D = torch.optim.Adam(D.parameters(), lr=0.0002)
    loss_func = nn.BCELoss()

    """每个epoch中，遍历训练数据集，调用训练函数进行训练，并计算总的判别器和生成器损失。
    在每个epoch的特定步数（每100个步骤或每个epoch结束时），打印当前的损失情况。"""
    for epoch in range(num_epoch): #逐个轮次训练，每一轮将使用全部的样本数据一次
        # 获取生成器 G 的参数
        generator_params = list(G.parameters())
        #print('g_para:\n',generator_params)
        # 获取判别器 D 的参数
        discriminator_params = list(D.parameters())
        #print('d_para:\n',discriminator_params)
        total_loss_D, total_loss_G = 0, 0
        """逐个批次的使用样本数据"""
        #每次迭代时，x将包含128个训练（真实）样本的张量。
        for i, (x, _) in enumerate(train_loader):  # i表示当前批次的索引，(x, _)表示当前批次的数据和标签，其中标签在这里用下划线(_)表示忽略。
            loss_D, loss_G = training(x) #训练一个批次的样本数据，返回loss，在training中更新了G和D

            total_loss_D += loss_D
            total_loss_G += loss_G
            #print('true_x:',x)

            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                print('Epoch {:02d} | Step {:04d} / {} | Loss_D {:.4f} | Loss_G {:.4f}'.format(epoch, i + 1,
                                                                                               len(train_loader),
                                                                                               total_loss_D / (i + 1),
                                                                                               total_loss_G / (i + 1)))

        """内嵌的for循环每次完整执行一次，就用所有数据把G和D完整更新一次，有了更新后的G和D"""
        y = torch.randn(batch_size, input_dim).to(device) #重新赋值G的输入噪声数据（样本数据）//表示生成一个形状为(64, input_dim)的张量，其中64表示批次大小，input_dim表示每个数据点的维度。有64个数据点，每个数据点都是一个噪声向量，用于生成对应的图像。
        img = G(y)#使用训练之后的G生成图片,是一个图片张量
        print('img:',img) #传入到G中forward函数中的x
        #print('G.forward(y):',G.forward(y))
        print('D(img)', torch.max(D(img)))#生成的一张图的张量里有64个批次，所以这个输出有64个概率。一个epoch的大循环生成64个概率
        save_image(img, './data/result/' + '%d_epoch.png' % epoch) #一个轮次保存一张图片---把img保存为'./data/result/' + '%d_epoch.png'图片
        #save_image 是一个函数，用于将张量保存为图像文件。它通常用于将生成器生成的图像保存到磁盘上。
