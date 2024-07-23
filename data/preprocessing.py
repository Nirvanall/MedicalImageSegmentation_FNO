#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 15:33:08 2023

Preprocess large binary images into smaller patches.
This code generates 256x256 images/masks.

@author: liulu
"""
import os
import cv2
from patchify import patchify

data_root = 'Fluo-N2DL-HeLa/02/'
label_root = 'Fluo-N2DL-HeLa/02_ST/SEG'

save_dir = 'Fluo-N2DL-HeLa-256/'

patch_size = (256,256)
patch_step = 128 # if patch_step=256, there is no overlap between patches


data_files = sorted([
            os.path.join(data_root, fname)
            for fname in os.listdir(data_root)
            if fname.endswith(".tif")
        ])
label_files = sorted([
            os.path.join(label_root, fname)
            for fname in os.listdir(label_root)
            if fname.endswith(".tif") and not fname.startswith(".")
        ])

for i in range(0, len(data_files)):
    image = cv2.imread(data_files[i],0)
    label = cv2.imread(label_files[i],0)
    image_patches = patchify(image, patch_size, step=patch_step)
    label_patches = patchify(label, patch_size, step=patch_step)
    for n in range(0, image_patches.shape[0]):
        for m in range(0, image_patches.shape[1]):
            label_patch = label_patches[n,m,:,:]
            if label_patch.max()>label_patch.min():
                image_patch = image_patches[n,m,:,:]
                cv2.imwrite(save_dir+'images/'+'_02_'+data_files[i][18:-4]+'_'+str(n)+'_'+str(m)+'.tif', image_patch)
                cv2.imwrite(save_dir+'labels/'+'_02_'+label_files[i][25:-4]+'_'+str(n)+'_'+str(m)+'.tif', label_patch)
                
                
    
    
