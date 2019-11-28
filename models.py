import torch
from config import *
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class GLU(nn.Module):
    # channel减半的一种玄学激活函数
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])



def conv1x1(in_planes, out_planes, bias = False):
   # '1x1 convolution with padding'
    return nn.Conv2d(in_planes, out_planes, 
        kernel_size = 1, stride=1, padding=0, bias = bias)

def conv3x3(in_planes, out_planes):
    # '3x3 convolution with padding'
    # 3x3的cnn，不改变形状
    return nn.Conv2d(in_planes, out_planes, 
        kernel_size = 3, stride=1, padding=1, bias = False)



#Upscale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),# width, height double
        conv3x3(in_planes, out_planes * 2),# channel变成out的2被，为后来的GLU提供原料
        nn.BatchNorm2d(out_planes  * 2),
        GLU())# half the channel
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out

def upSamples(channel_num, times):
    # 按照需求扩大图像
    upsamples = nn.Sequential()
    for i in range(times):
        name = 'up_sample' + str(i)
        upsamples.add_module(name, upBlock(channel_num, int(channel_num/2)))
        channel_num = int(channel_num/2)
    return upsamples

class Init_generate_stage(nn.Module):
    # 输入一个向量，把向量用fc展开，
    def __init__(self, condition_dim, input_dim):
        # 128, 100
        super(Init_generate_stage, self).__init__()

        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.define_module()

    def define_module(self):
        # 在这里，我们要输出的是一个128 * 64 * 64的东西
        
        self.full_connect = nn.Sequential(
            nn.Linear(self.input_dim, self.condition_dim * 16 * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(self.condition_dim * 16 * 4 * 4 * 2),
            GLU()
        )
        # 128 x 64 x 64的
        # 我们在这里会扩展为128*16 x 4 x 4
        self.upSamples = upSamples(self.condition_dim * 16, 4)

    def forward(self, input, noise):
        real_input = torch.cat((input, noise), 1)
        #print(real_input.shape, self.input_dim)
        output = self.full_connect(real_input)
        output = output.view(-1, self.condition_dim * 16, 4, 4)
        output = self.upSamples(output) 
        return output
        

class Generate_image(nn.Module):
    def __init__(self, condition_dim):
        super(Generate_image, self).__init__()
        self.condition_dim = condition_dim
        self.image_net = nn.Sequential(
            conv3x3(condition_dim, 3),
            nn.Tanh()
        )
    def forward(self, image_condition):
        out_image = self.image_net(image_condition)
        
        return out_image

class Next_generate_stage(nn.Module):
    # 这里我想要尝试一下，如果每一层都重新考虑一次句子的内容，会不会有所改善呢
    def __init__(self, condition_dim, input_dim, text_dim):
        super(Next_generate_stage, self).__init__()
        self.condition_dim = condition_dim
        self.input_dim = input_dim
        self.text_dim = text_dim
        self.define_module()
        self.define_ResNet()
        self.combine_layer = nn.Sequential(
            conv3x3(condition_dim + text_dim, condition_dim * 4),
            nn.BatchNorm2d(condition_dim * 4, condition_dim * 4),
            GLU()
        )
        self.up_sample = upSamples(condition_dim * 2, 1)

    def define_ResNet(self):
        self.resnet = nn.Sequential()
        for i in range(RESNET_LAYER):
            name = 'resnet' + str(i)
            self.resnet.add_module(name, ResBlock(self.condition_dim * 2))
        

    def define_module(self):
        # 在这里，我们要输出的是一个128 * 64 * 64的东西
        self.full_connect = nn.Sequential(
            nn.Linear(self.input_dim, self.condition_dim * 16 * 4 * 4 * 2, bias = False),
            nn.BatchNorm1d(self.condition_dim * 16 * 4 * 4 * 2),
            GLU()
        )
        # 128 x 64 x 64的
        # 我们在这里会扩展为128*16 x 4 x 4
        condition_dim = self.condition_dim
        #print(self.text_dim)
        expend_times = int(np.log2(128 * 128 / self.text_dim) - 3)
        
        #print(expend_times)
        self.up_samples = upSamples(self.condition_dim * 16, expend_times)

    def forward(self, input, raw_image):
        # 首先我们先拿到text
        text_info = self.full_connect(input)

        #image_size = np.log2(raw_image.shape[2]) # 我们拿到图片的一个边
        #double_times = image_size - 2# -2是因为我们从一开始的图像就已经是4x4了， double_times表示我们要让这个矩阵的大小翻倍的次数
        double_times = 16 # 我思考了一下，似乎不应该让文本在后面的部分占太多比重，所以还是改成固定的吧
        text_info = text_info.view(-1, self.condition_dim * double_times, 4, 4)
        
        text_info = self.up_samples(text_info) 
        
        # 到这里，我们已经得到了一个128 * size * size的一个数组，可以和raw_image拼一起扔到Resnet里面搞事情了
        #print(text_info.shape, raw_image.shape)
        real_input =  torch.cat((text_info, raw_image), 1)
        #print(real_input.shape, 'world')
        output =  self.combine_layer(real_input)
        #print(output.shape, 'so far so good')
        output = self.resnet(output)
        #print(output.shape, 'so far so good')
        output = self.up_sample(output)
        #print(output.shape, 'so far so good')
        return output



def con3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace = True)
    )
    return block

class D_LOGITS(nn.Module):
    #在这里我们读入一个text_embed,以及一个处理过的feature512 x 4 x 4
    # 我们这样思考一下，就是不管多大，先包装到4x4，然后用conv2d降维
    def __init__(self, bcondition = False):
        super(D_LOGITS, self).__init__()
        self.bcondition = bcondition
        self.out_logits = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4, stride = 4),
            nn.Sigmoid()
        )
        if self.bcondition:
            self.jointConv = con3x3_leakRelu(512 + 256, 512)

    def forward(self, image_code, text_embed = None):
        if self.bcondition and text_embed is not None:
            text_embed = text_embed.view(-1, 256, 1, 1)
            text_embed = text_embed.repeat(1, 1, 4, 4)
            mix_code = torch.cat((image_code, text_embed), 1)
            mix_code = self.jointConv(mix_code)
        else:
            mix_code = image_code
        mix_code = self.out_logits(mix_code)
        return mix_code.view(-1)



        

class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.image_encode_net = encode_image(512, 4)
        self.uncondition_DNET = D_LOGITS()
        self.condition_DNET = D_LOGITS(True)

    def forward(self, image):
        return self.image_encode_net(image)

class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.image_encode_net = encode_image(512, 5)
        self.uncondition_DNET = D_LOGITS()
        self.condition_DNET = D_LOGITS(True)
    def forward(self, image):
        return self.image_encode_net(image)


class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.image_encode_net = encode_image(512, 6)
        self.uncondition_DNET = D_LOGITS()
        self.condition_DNET = D_LOGITS(True)
    def forward(self, image):
        return self.image_encode_net(image)



def encode_image(base_size, encode_time):
    block = nn.Sequential(downBlock(3, base_size))
    for i in range(encode_time - 1):
        block.add_module('downblock' + str(i), downBlock(base_size, base_size))
    return block

def downBlock(input_channel, output_channel):
    block = nn.Sequential(
        nn.Conv2d(input_channel, output_channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
        nn.BatchNorm2d(output_channel),
        nn.LeakyReLU(0.2, inplace= True)
    )
    return block

class CA_NET(nn.Module):
    # 因为我们的latency condition有很多维，但是数据却不够，导致训练的模型会出现不连续性，很多情况顾及不到
    # 所以我们引入了一个数据增强的机制，增强网络的鲁棒性
    def __init__(self):
        super(CA_NET, self).__init__()
        self.sent_embed_dim = 256
        self.condition = 100
        self.fc = nn.Linear(self.sent_embed_dim, self.condition * 4, bias = True)
        self.relu = GLU()

    def encode(self, sent_embedding):
        # 400 -> 200
        x = self.relu(self.fc(sent_embedding))
        mu = x[:, :self.condition]
        logvar = x[:, self.condition:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        # 这里就是数据增强的部分，eps引入随机噪声
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, sent_embedding):
        mu, logvar = self.encode(sent_embedding)
        condition_code = self.reparametrize(mu, logvar)
        return condition_code


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.ca_net = CA_NET()
        self.stage1_g = Init_generate_stage(128, 200)
        self.stage2_g = Next_generate_stage(128, 100, 128)
        self.stage3_g = Next_generate_stage(128, 100, 64)
        self.gen1 = Generate_image(128)
        self.gen2 = Generate_image(128)
        self.gen3 = Generate_image(128)
    
    def forward(self, noise, sent_embs):
        
        condition_code = self.ca_net(sent_embs)
        #print(type(condition_code), type(noise))
        image_condition1 = self.stage1_g(condition_code, noise)
        image_condition2 = self.stage2_g(condition_code, image_condition1)
        image_condition3 = self.stage3_g(condition_code, image_condition2)
        fake_images = []
        fake_images.append(self.gen1(image_condition1))
        fake_images.append(self.gen2(image_condition2))
        fake_images.append(self.gen3(image_condition3))
        return fake_images