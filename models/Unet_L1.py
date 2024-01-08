import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class PairModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize
        self.opt = opt
        self.input_A = self.Tensor(nb, opt.input_nc, size, size)
        self.input_B = self.Tensor(nb, opt.output_nc, size, size)
        self.input_img = self.Tensor(nb, opt.input_nc, size, size)
        self.input_A_gray = self.Tensor(nb, 1, size, size)

        if opt.vgg > 0:
            self.vgg_loss = networks.PerceptualLoss()
            self.vgg_loss.cuda()
            self.vgg = networks.load_vgg16("./model")
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        skip = True if opt.skip > 0 else False
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, skip=skip, opt=opt)
            #opt.input_nc：输入图像的通道数。
            #opt.output_nc：输出图像的通道数。
            #opt.ngf：生成器网络中的特征图的基础通道数。
            #opt.which_model_netG：生成器网络的类型。
            #opt.norm：归一化的类型。
            #not opt.no_dropout：是否使用 dropout 层。
            #self.gpu_ids：GPU 的 ID 列表，用于将生成器网络移动到对应的 GPU 上。
            #opt=opt：选项对象，包含其他的参数和设置
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG_A, 'G_A', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_wgan:
                self.criterionGAN = networks.DiscLossWGANGP()
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            if opt.use_mse:
                self.criterionCycle = torch.nn.MSELoss()
            else:
                self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        if opt.isTrain:
            self.netG_A.train()
        else:
            self.netG_A.eval()
        print('-----------------------------------------------')

    def set_input(self, input): #设置输入
        AtoB = self.opt.which_direction == 'AtoB' #参数判断是否为 A 到 B 的转换
        input_A = input['A' if AtoB else 'B'] #根据 AtoB 的值选择对应的输入图像
        input_B = input['B' if AtoB else 'A'] #根据 AtoB 的值选择对应的输入图像
        input_img = input['input_img'] #输入图像
        input_A_gray = input['A_gray'] #输入灰度图
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_A_gray.resize_(input_A_gray.size()).copy_(input_A_gray)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.image_paths = input['A_paths' if AtoB else 'B_paths'] #保存输入图像路径

    def forward(self): #前向传递
        self.real_A = Variable(self.input_A) #将输入图像 A 转换为可变张量，并赋值给 self.real_A
        self.real_B = Variable(self.input_B) #将输入图像 B 转换为可变张量，并赋值给 self.real_B
        self.real_A_gray = Variable(self.input_A_gray)
        self.real_img = Variable(self.input_img)


    def test(self):
        self.real_A = Variable(self.input_A, volatile=True) #将输入图像 A 转换为只读的可变张量，并赋值给 self.real_A
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray) #通过生成器网络 netG_A 对输入图像 A 进行前向传递，生成假图像 B 和潜在的 real_A

        self.real_B = Variable(self.input_B, volatile=True)

    def predict(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray) #通过生成器网络 netG_A 对输入图像 A 进行前向传递，生成假图像 B 和潜在的 real_A

        real_A = util.tensor2im(self.real_A.data) #将 self.real_A 转换为可视化图像的数据格式并赋值给 real_A
        fake_B = util.tensor2im(self.fake_B.data)
        if self.opt.skip == 1:
            latent_real_A = util.tensor2im(self.latent_real_A.data) #将 self.latent_real_A 转换为可视化图像的数据格式并赋值给 latent_real_A
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ("latent_real_A", latent_real_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_G(self):

        self.fake_B, self.latent_real_A = self.netG_A.forward(self.real_A, self.real_A_gray) #通过生成器网络 netG_A 对输入图像 A 进行前向传递，生成假图像 B 和潜在的 real_A
         # = self.latent_real_A + self.opt.skip * self.real_A
        self.L1_AB = self.criterionL1(self.fake_B, self.real_B) * self.opt.l1 #计算假图像 B 与真实图像 B 之间的 L1 损失，并乘以 opt.l1 权重
        self.loss_G = self.L1_AB
        self.loss_G.backward() #更新网络参数


    def optimize_parameters(self, epoch): #优化
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


    def get_current_errors(self, epoch): #获取当前训练状态的损失值
        L1 = self.L1_AB.data[0] #从 L1 损失张量 self.L1_AB 中获取数值，并赋值给 L1
        loss_G = self.loss_G.data[0] #从总损失张量 self.loss_G 中获取数值，并赋值给 loss_G
        return OrderedDict([('L1', L1), ('loss_G', loss_G)])

    def get_current_visuals(self): #获取当前生成的图像
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
#保存模型和更新学习率
    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)

    def update_learning_rate(self):
        
        if self.opt.new_lr:
            lr = self.old_lr/2
        else:
            lrd = self.opt.lr / self.opt.niter_decay
            lr = self.old_lr - lrd
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
