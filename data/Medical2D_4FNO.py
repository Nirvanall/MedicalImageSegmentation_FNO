#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:07:37 2023

load 2D data

@author: liulu
"""
import torch
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from neuralop.utils import UnitGaussianNormalizer


class Medical2D_4FNO(Dataset):
    def __init__(self, data_root, label_root,
                 sample_size = (32,32),
                 encode_input=False,
                 encode_output=True,
                 encoding='channel-wise',
                 transform_x=None,
                 transform_y=None):
        self.data_root = data_root
        self.label_root = label_root
        self.sample_size = sample_size
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.encoding = encoding
        self.output_encoder = None
        self.transform_x = transform_x
        self.transform_y = transform_y
        
        self.data_files = sorted([
            os.path.join(data_root, fname)
            for fname in os.listdir(data_root)
            if fname.endswith(('.png', '.tif', 'jpg'))
        ])
        self.label_files = sorted([
            os.path.join(label_root, fname)
            for fname in os.listdir(label_root)
            if fname.endswith(('.png', '.tif', 'jpg')) and not fname.startswith(".")
        ])

    def __len__(self):
      return len(self.label_files)

    def __getitem__(self, index):
       """Return tuple (input, target)"""
       data_path = self.data_files[index]
       label_path = self.label_files[index]

       data_image = Image.open(data_path)
       #data_image = data_image.convert("L") # convert to grayscale
       label_image = Image.open(label_path)
       
       data_image = np.array(data_image, dtype='uint8')
       label_image = np.array(label_image, dtype='uint8')
       
       data_transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Resize(self.sample_size),
           transforms.Normalize((0.5,), (0.5,))
           ])
       data_image = data_transform(data_image)
       
       label_transform = transforms.Compose([
          transforms.ToTensor(),
          transforms.Resize(self.sample_size)
       ])
       label_tensor = label_transform(label_image).to(torch.float32)
       
       if self.encode_input:
           if self.encoding == 'channel-wise':
               reduce_dims = list(range(data_image.ndim))
           elif self.encoding == 'pixel-wise':
               reduce_dims =[0]
           input_encoder = UnitGaussianNormalizer(data_image, reduce_dim=reduce_dims,verbose=False)
           data_image = input_encoder.encode(data_image)
       else:
           input_encoder = None
    
       if self.encode_output:
           if self.encoding == 'channel-wise':
               reduce_dims = list(range(label_tensor.ndim))
           elif self.encoding == 'pixel-wise':
               reduce_dims = [0]
               
           self.output_encoder=UnitGaussianNormalizer(label_tensor, reduce_dim=reduce_dims,verbose=False)
           label_tensor = self.output_encoder.encode(label_tensor)
       else:
           self.output_encoder = None

       if self.transform_x is not None:
           data_image = self.transform_x(data_image)

       if self.transform_y is not None:
           label_tensor = self.transform_y(label_tensor)
        

       return {'x': data_image, 'y': label_tensor}
   
    def get_output_encoder(self):
        return self.output_encoder
