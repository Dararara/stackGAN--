import numpy as np

import torchvision.transforms as transforms
from PIL import Image
import os, sys
from torch.utils.data import Dataset, DataLoader
import torch
from config import *
import torch.nn as nn
from collections import defaultdict
import re
import torch.optim as optim
from torch.autograd import Variable

from models import Init_generate_stage, Generate_image, Next_generate_stage
from models import encode_image
from dataloader import TextDataset
from code_copy import RNN_ENCODER
from models import D_NET64, D_NET128, D_NET256, G_NET
from loss import discriminator_loss, generator_loss
def define_optimizers(netG, netDs):
    optimizersD = []
    for i in range(len(netDs)):
        opt = optim.Adam(netDs[i].parameters(),
            lr = 0.001,
            betas=(0.5, 0.999)
        )
        optimizersD.append(opt)
        #print(opt)
        #for para in netDs[i].parameters():
        #    print(para.size())
        
    optimizerG = optim.Adam(netG.parameters(), 
        lr = 0.001,
        betas=(0.5, 0.999)
    )
    return optimizerG, optimizersD

def save_single_image(image, save_path):
    pass


torch.cuda.set_device(0)
block = encode_image(512, 4)

data = TextDataset(image_path, text_path)
dataloader = DataLoader(data, batch_size=2, shuffle=True)




dis1 = D_NET64().cuda()
dis2 = D_NET128().cuda()
dis3 = D_NET256().cuda()


g_net = G_NET().cuda()
d_nets = [dis1, dis2, dis3]

optimizerG, optimizersD = define_optimizers(g_net, d_nets)
# 这里非常恶心，卡得很死，日后最好能够自己训练一个出来
text_encoder = RNN_ENCODER(data.word_count, nhidden=256)
text_encoder_path = 'model/text_encoder200.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from: ', text_encoder_path)
text_encoder.eval()
text_encoder = text_encoder.cuda()


for p in d_nets[0].parameters():
    print(p.size())


batch_size = 2
noise = Variable(torch.FloatTensor(batch_size, 100)).cuda()

for epoch in range(100):
    total_error_g = 0
    total_error_d = 0
    dataiter = iter(dataloader)
    for iiii in range(int(data.len/batch_size)):
        noise.data.normal_(0, 1)
        print("start load")
        text, images, text_len = dataiter.next()
        print('load finish')        
        text_len, indices = torch.sort(text_len, 0, True)
        text = text[indices]
        text = Variable(text).cuda()
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1)).cuda()
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0)).cuda()

        for i in range(len(images)):
            images[i] = Variable(images[i]).cuda()
        text_len = Variable(text_len).cuda()
        #print(text.shape, text_len)
        
        hidden = text_encoder.init_hidden(2)
        words_embs, sent_embs = text_encoder(text, text_len, hidden)
        sent_embs = sent_embs.detach()
        real_labels = real_labels.detach()
        fake_labels = fake_labels.detach()
        

        fake_images = g_net(noise, sent_embs)
        
        for i in range(len(d_nets)):
            d_nets[i].zero_grad()
            errD = discriminator_loss(d_nets[i], images[i].detach(), fake_images[i].detach(), sent_embs, real_labels.detach(), fake_labels.detach())
            errD.backward()
            optimizersD[i].step()
            total_error_d += errD
        
        g_net.zero_grad()
        errG_total = generator_loss(d_nets, fake_images, sent_embs, real_labels)
        errG_total.backward()
        optimizerG.step()
        total_error_g += errG_total
    print('total error: g: ', total_error_g, ' d: ', total_error_d)
    save_single_image(fake_images[2], 'fake' + str(epoch) + '.png')
    

torch.save(g_net, 'G_NET.pth')
for i in range(len(d_nets)):
    torch.save(d_nets[i].state_dict(), 'D_NET' + str(i) + '.pth')



    '''
    i = 2
    d_nets[i].zero_grad()
    errD1 = discriminator_loss(d_nets[i], images[i], fake_images[i], sent_embs, real_labels, fake_labels)
    errD1.backward()
    optimizersD[i].step()

    '''




    
'''
    # 就很绝望，搞不定啊，GPU显存不够用，哭了
    b = stage1_g(sent_embs, sent_embs)
    print(b.shape)
    c = gen1(b)
    print(c.shape)
    label1 = dis1(c)
    print('label1 ', label1.shape)
    print(dis1.uncondition_DNET(label1).shape)
    print(dis1.condition_DNET(label1, sent_embs).shape)


    d = stage2_g(sent_embs, b)
    print(d.shape)
    print(gen2(d).shape)
    label2 = dis2(gen2(d))
    print('label2 ', label2.shape)
    print(dis2.uncondition_DNET(label2).shape)
    print(dis2.condition_DNET(label2, sent_embs).shape)
    
    e = stage3_g(sent_embs, d)
    print(e.shape)
    print(gen3(e).shape)
    label3 = dis3(gen3(e))
    print('label3 ', label3.shape)
    print(dis3.uncondition_DNET(label3).shape)
    print(dis3.condition_DNET(label3, sent_embs).shape)
'''