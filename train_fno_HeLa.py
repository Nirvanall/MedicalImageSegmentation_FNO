#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:06:28 2023

train FNO with HeLa data
We use 64x64 images for training

@author: liulu
"""

from data.Medical2D_4FNO import Medical2D_4FNO
import torch
import sys
from torch.utils.data import DataLoader
import numpy as np
from datetime import date
from neuralop.models import TFNO
from neuralop import Trainer, LpLoss, H1Loss
from neuralop.utils import count_params
from neuralop.datasets.transforms import PositionalEmbedding

today = date.today()
save_name = 'HeLa64-TFNO-32-64-128-' + str(today.strftime('%d%m%Y'))
input_dir = "HeLa/Fluo-N2DL-HeLa-256-tif/images"
target_dir = "HeLa/Fluo-N2DL-HeLa-256-tif/labels"
sample_size = (64,64)
batch_size = 16
seed = 42

if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = "cpu"
  
grid_boundaries=[[0,1],[0,1]]
positional_encoding=True

data = Medical2D_4FNO(input_dir, target_dir, sample_size=sample_size, transform_x=PositionalEmbedding(grid_boundaries, 0) if positional_encoding else None)
generator=torch.Generator().manual_seed(seed)
train, valid = torch.utils.data.random_split(data, [4600, 552])
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
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
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

#%% plot training loss curve
import matplotlib.pyplot as plt
training_loss_dict = trainer.get_training_loss()
fig = plt.figure()
plt.plot(list(training_loss_dict.keys()), list(training_loss_dict.values()))
plt.xlabel('#epoches')
plt.ylabel('loss')
plt.title('Training loss Curve')
fig.savefig(save_name + '-training-loss.png')
fig.show()

#%% load the best checkpoint
model_path = save_name + '-best-checkpoint.pt'
#model.load_state_dict(torch.load(model_path))
model = torch.load(model_path, map_location=device)

#%% plot the prediction
#from skimage.filters import threshold_otsu

test_samples = valid_loader.dataset

fig = plt.figure(figsize=(7, 20))
i = [0, 10, 20, 30,40, 50,60,70,80,90]
for index in range(5):
    data = test_samples[i[index]]
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
    #thres = threshold_otsu(y.squeeze().detach().cpu().numpy())
    #ax.imshow(y.squeeze()>thres)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(10, 3, index*3 + 3)
    #thres = threshold_otsu(out.squeeze().detach().cpu().numpy())
    ax.imshow(out.squeeze().detach().cpu().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

plt.tight_layout()
fig.show()
fig.savefig(save_name + '-prediction.png')



#%% evaluation/test with dice, miou, sensitivity
'''
from torchmetrics.classification import BinaryJaccardIndex, BinaryF1Score, BinaryRecall
from skimage.filters import threshold_otsu

jaccard = BinaryJaccardIndex()
dice = BinaryF1Score()
sensitivity = BinaryRecall()

test_samples = valid_loader.dataset
iou = []
dsc = []
recall = []
for data in test_samples:
    x = data['x']
    y = data['y']
    thres_y = threshold_otsu(y.squeeze().detach().cpu().numpy())
    out =model(x.unsqueeze(0).to(device))
    thres_out = threshold_otsu(out.squeeze().detach().cpu().numpy())
    
    
    iou.append(jaccard(out.squeeze()>thres_out, y.squeeze()>thres_y).item()) 
    dsc.append(dice(out.squeeze()>thres_out, y.squeeze()>thres_y).item())
    recall.append(sensitivity(out.squeeze()>thres_out, y.squeeze()>thres_y).item())

print('>> Testing dataset mIoU = {:.4f}'.format(np.mean(iou))+'±{:.4f}'.format(np.std(iou)))
print('>> Testing dataset DSC = {:.4f}'.format(np.mean(dsc))+'±{:.4f}'.format(np.std(dsc)))
print('>> Testing dataset Sensitivity = {:.4f}'.format(np.mean(recall))+'±{:.4f}'.format(np.std(recall)))
    
'''
