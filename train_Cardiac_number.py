#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 17:06:48 2023
train FNO

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
           transforms.Resize((64,64)),
           transforms.ToTensor(),
           transforms.Normalize((0.5,), (0.5,))
           ])
       data_image = data_transform(data_image)
       

       label_transform = transforms.Compose([
          transforms.Resize((64,64)),
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
save_name = 'CardiacFat64_25-ReduceLROnPlateau-TFNO-32-64-128-' + str(today.strftime('%d%m%Y'))
input_dir = "../fat_png_normal/training/input/"
target_dir = "../fat_png_normal/training/output/"
test_input_dir = "../fat_png_normal/testing/input/"
test_output_dir = "../fat_png_normal/testing/output/"
img_size = (64,64)
num_classes = 2
batch_size = 16
seed = 42

if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = "cpu"
  
grid_boundaries=[[0,1],[0,1]]
positional_encoding=True


data = CardiacFat(input_dir, target_dir, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
valid = CardiacFat(test_input_dir, test_output_dir, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
generator=torch.Generator().manual_seed(seed)
#train, valid = torch.utils.data.random_split(data, [7000, 367], generator=generator)
#train, valid, left = torch.utils.data.random_split(data, [400,20, 6970], generator=generator)
train, left = torch.utils.data.random_split(data, [25, 8975], generator=generator)
output_encoder = data.get_output_encoder()

train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0, persistent_workers=False)
valid_loader = DataLoader(valid, batch_size=batch_size, pin_memory=True, num_workers=0, persistent_workers=False)

#%%
model = TFNO(n_modes=(32,32), hidden_channels=64, in_channels=3, out_channels=1, projection_channels=128, factorization='tucker', rank=0.42)
model = model.to(device)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=2e-3,
                                weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=10)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%


print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model, n_epochs=300,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=1,
                  use_distributed=False,
                  verbose=True,
                  save_name=save_name)

trainer.train(train_loader, valid_loader,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

#%% load the best checkpoint
model_path = save_name + '-best-checkpoint.pt'
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path, map_location=device)

#%% plot the prediction
import matplotlib.pyplot as plt
#%% plot training loss curve
training_loss_dict = trainer.get_training_loss()
fig = plt.figure()
plt.plot(list(training_loss_dict.keys()), list(training_loss_dict.values()))
plt.xlabel('#epoches')
plt.ylabel('loss')
plt.title('Training loss Curve')
fig.savefig(save_name + '-training-loss.png')
fig.show()

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
    ax.imshow(y.squeeze())
    if index == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(10, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().cpu().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

#fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
fig.savefig(save_name + '-prediction.png')


