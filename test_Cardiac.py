#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:06:48 2023
test FNO

@author: liulu
"""

import torch
import sys
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from datetime import date

from neuralop.models import TFNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.utils import UnitGaussianNormalizer, count_params
from neuralop.datasets.transforms import PositionalEmbedding
import matplotlib.pyplot as plt
import skimage
from skimage.filters import threshold_otsu

class CardiacFat(Dataset):
    def __init__(self, data_root, label_root,
                 encode_input=False,
                 encode_output=True,
                 encoding='channel-wise',
                 transform_x=None,
                 transform_y=None):
        self.data_root = data_root
        self.label_root = label_root
        self.encode_input = encode_input
        self.encode_output = encode_output
        self.encoding = encoding
        self.output_encoder = None
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.data_files = sorted([
            os.path.join(data_root, fname)
            for fname in os.listdir(data_root)
            if fname.endswith(".png")
        ])
        self.label_files = sorted([
            os.path.join(label_root, fname)
            for fname in os.listdir(label_root)
            if fname.endswith(".png") and not fname.startswith(".")
        ])

    def __len__(self):
      return len(self.label_files)

    def __getitem__(self, index):
       """Return tuple (input, target)"""
       data_path = self.data_files[index]
       label_path = self.label_files[index]

       data_image = Image.open(data_path)
       data_image = data_image.convert("L") # convert to grayscale
       label_image = Image.open(label_path)
       
       data_transform = transforms.Compose([
           transforms.Resize((256,256)),
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
           ])
       data_image = data_transform(data_image)
       
       label_transform = transforms.Compose([
          transforms.Resize((256,256)),
          transforms.PILToTensor()
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
   
#%%
today = date.today()
save_name = 'CardiacFat64-256-ReduceLROnPlateau-TFNO-32-64-128-' + str(today.strftime('%d%m%Y'))
input_dir = "../fat_png_normal/training/input/"
target_dir = "../fat_png_normal/training/output/"
test_input_dir = "../fat_png_normal/testing/input/"
test_output_dir = "../fat_png_normal/testing/output/"
num_classes = 2
batch_size = 16
seed = 42

device = "cpu"
  
grid_boundaries=[[0,1],[0,1]]
positional_encoding=True


data = CardiacFat(input_dir, target_dir, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
valid = CardiacFat(test_input_dir, test_output_dir, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
generator=torch.Generator().manual_seed(seed)
output_encoder = data.get_output_encoder()

train_loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, persistent_workers=False)
valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=True, num_workers=0, persistent_workers=False)


#%% load the best checkpoint
model_path = 'CardiacFat64-ReduceLROnPlateau-TFNO-32-64-128-23122023-best-checkpoint.pt'
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path, map_location=device)

#%% plot the prediction

test_samples = valid_loader.dataset

fig = plt.figure(figsize=(7, 20))
for index in range(10):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0).to(device))

    ax = fig.add_subplot(10, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(10, 3, index*3 + 2)
    thres = threshold_otsu(y.squeeze().detach().cpu().numpy())
    ax.imshow(y.squeeze()>thres)
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(10, 3, index*3 + 3)
    thres = threshold_otsu(out.squeeze().detach().cpu().numpy())
    ax.imshow(out.squeeze().detach().cpu().numpy()> thres)
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

#fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig(save_name + '-prediction.png')


