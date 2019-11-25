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
from torch.autograd import Variable

from models import Init_generate_stage, Generate_image, Next_generate_stage
from models import encode_image
from dataloader import TextDataset
from code_copy import RNN_ENCODER
from models import D_NET64, D_NET128, D_NET256



block = encode_image(512, 4)

data = TextDataset(image_path, text_path)
dataloader = DataLoader(data, batch_size=2, shuffle=True)
dataiter = iter(dataloader)


stage1_g = Init_generate_stage(128, 512).cuda()
gen1 = Generate_image(128).cuda()
stage2_g = Next_generate_stage(128, 256, 128).cuda()
stage3_g = Next_generate_stage(128, 256, 64).cuda()
gen2 = Generate_image(128).cuda()
gen3 = Generate_image(128).cuda()
dis1 = D_NET64().cuda()
dis2 = D_NET128().cuda()
dis3 = D_NET256().cuda()


# 这里非常恶心，卡得很死，日后最好能够自己训练一个出来
text_encoder = RNN_ENCODER(data.word_count, nhidden=256)
text_encoder_path = 'model/text_encoder200.pth'
state_dict = torch.load(text_encoder_path, map_location=lambda storage, loc: storage)
text_encoder.load_state_dict(state_dict)
for p in text_encoder.parameters():
    p.requires_grad = False
print('Load text encoder from: ', text_encoder_path)
text_encoder.eval()


for i in range(2):

    text, images, text_len = dataiter.next()
    


    text_len, indices = torch.sort(text_len, 0, True)


    text = text[indices]

    hidden = text_encoder.init_hidden(2)
    words_embs, sent_embs = text_encoder(text, text_len, hidden)
    
    text = Variable(text).cuda()
    for i in range(len(images)):
        images[i] = Variable(images[i]).cuda()
    text_len = Variable(text_len).cuda()
    stage1_g = stage1_g.cuda()
    print(text.shape, text_len)
    sent_embs = sent_embs.cuda()

    # 就很绝望，搞不定啊，GPU显存不够用，哭了
    b = stage1_g(sent_embs, sent_embs)
    print(b.shape)
    c = gen1(b)
    print(c.shape)
    label1 = dis1(c)
    print('label1 ', label1.shape)
    print(dis1.uncondition_DNET(label1).shape)
    print(dis1.condition_DNET(label1, sent_embs).shape)

'''
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