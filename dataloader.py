import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import os, sys
from torch.utils.data import Dataset, DataLoader
import torch
from config import *
import torch.nn as nn
import torch.nn.functional as F
from models import *
from collections import defaultdict
import re

class TextDataset(Dataset):
    def __init__(self, image_path, text_path):
        self.image_path = image_path
        self.text_path = text_path


        self.image_transform = transforms.Compose([
            transforms.Resize(int(image_size * 76 / 64)), # 得到一个比目标稍大的图片
            transforms.RandomCrop(image_size), # 随机选取其中一部分
            transforms.RandomHorizontalFlip()# 随机进行水平翻转
        ])

        # 对图片进行归一化操作
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        folder_names = os.listdir(image_path)
        self.file_names = []
        for folder in folder_names:
            files = os.listdir(os.path.join(image_path, folder))
            #print(len(files))
            for file in files:
                file = os.path.join(folder, file)
                self.file_names.append(file)
        self.len = len(self.file_names)
        self.text_load()

    def text_load(self):
        index_to_word = defaultdict(str)
        word_to_index = defaultdict(int)
        word_to_index['<end>'] = 0
        index_to_word[0] = '<end>'
        word_count = 0
        self.texts = []
        # 存储所有的文本描述
        for file in self.file_names:
            text_pos = os.path.join(self.text_path, file.replace('jpg', 'txt'))
            f = open(text_pos)
            text = f.readlines()
            f.close()
            # 存储对某个图像的文本描述
            sentences = []
            for i in text:
                # 这是一句话
                wordList = re.sub("[^\w]", " ",  i).split()
                sentence = []
                for word in wordList:
                    word = word.lower()
                    if word_to_index[word] == 0:
                        word_count += 1
                        word_to_index[word] = word_count
                        index_to_word[word_count] = word
                    sentence.append(word_to_index[word])
                sentences.append(sentence)
            self.texts.append(sentences)
        #print(word_to_index['hello'])
        #print(self.texts)
        self.word_count = len(word_to_index)
        self.texts = np.asarray(self.texts)


    def __getitem__(self, index):
        # 读取十句描述
        # 读取图片并做统一化处理
        img = Image.open(os.path.join(self.image_path, self.file_names[index]))
        img = self.image_transform(img)
        
        
        imgs = []
        size = [64, 128, 256]
        for i in size:
            resize_img = transforms.Resize(i)(img)
            resize_img = self.norm(resize_img)
            imgs.append(np.array(resize_img))
        
        #print(type(self.texts))
        text = self.texts[index][0]
        text_len = len(text)
        text_len = min(text_len, 20)
        while(len(text) < 20):
            text.append(0)
        text = text[:20]
        text = torch.LongTensor(text)
        
        return text, imgs, text_len
    
    def __len__(self):
        return 1000





