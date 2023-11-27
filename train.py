import gan
from gan import *
import torch
D_loss=[]
G_loss=[]
"""开始训练"""
for epoch in range(30):
    D_epoch_loss=0
    G_epoch_loss=0
    count=len(train_dl)
    for step,(img,_) in enumerate(train_dl):
        img=img.to(device)
        size=img.shape[0]
        random_seed=torch.randn(size,100,device=device) #生成随即输入
        d_optimizer.zero_grad()
        real_output= gan.dis(img)#判别器输入真实图片
        d_real_loss= gan.loss_fn(real_output,torch.ones_like(real_output,device=device))
        d_real_loss.backward()
        #生成器输入随即张量得到生成图片
        generated_img=gen(random_seed)
        #判别器输入生成图像
        fake_output=dis(generated_img.detach())
        d_fake_loss=loss_fn(fake_output,torch.zeros_like(fake_output,device=device))

        disc_loss=d_real_loss+d_fake_loss #判别器总模块
        fake_output=dis(generated_img)
        gen_loss=loss_fn(fake_output,torch.ones_like(fake_output,device=device))
        gen_loss.backward()#将损失loss向输入测进行反向传播，同时对需要进行梯度计算的所有变量计算梯度，并进行累积
        g_optimizer.step()#优化器对参数进行更新

        with torch.no_grad():
            D_epoch_loss+=disc_loss.item()
            G_epoch_loss+=gen_loss.item()
    with torch.no_grad():
        D_epoch_loss/=count
        G_epoch_loss/=count
        D_loss.append(D_epoch_loss)
        G_loss.append(G_epoch_loss)
        #训练完一个epoch，输出提示并绘制生成的图片
        print("Epoch:",epoch)
        generate_and_save_images(gen,epoch,test_seed)
