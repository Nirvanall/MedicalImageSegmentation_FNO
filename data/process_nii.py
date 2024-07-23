#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 16:23:24 2023

read .nii.gz files

@author: liulu
"""
from __future__ import print_function
import numpy as np
import os
import glob
import nibabel as nib
import skimage

def normalize(X):
    X = X - np.min(X)
    X = X/np.max(X)
    return X

data_dir= 'Task02_Heart/imagesTr'
label_dir = 'Task02_Heart/labelsTr'


data_path = glob.glob(os.path.join(data_dir, '*.nii.gz'))
label_path =  glob.glob(os.path.join(label_dir, '*.nii.gz'))

data_path = sorted(data_path)
label_path = sorted(label_path)

for i in range(0, len(data_path)):
    img = nib.load(data_path[i])
    img_array = np.array(img.dataobj)
    img_array = normalize(img_array)
    
    label = nib.load(label_path[i])
    label_array = np.array(label.dataobj)
    label_array = normalize(label_array)
    
    for j in range(0, img_array.shape[2]):
        label2d = label_array[:,:,j]
        img2d = img_array[:,:,j]
        skimage.io.imsave('Task02_Heart/imagesTr_png_full' + data_path[i][21:-7]+ '_' + str(j) + '.png', img2d)
        skimage.io.imsave('Task02_Heart/labelsTr_png_full' + label_path[i][21:-7]+ '_' + str(j) + '.png', label2d)
        '''
        if label2d.max()>label2d.min():
            img2d = img_array[:,:,j]
            skimage.io.imsave('Task09_Spleen/imagesTr_png' + data_path[i][22:-7]+ '_' + str(j) + '.png', img2d)
            skimage.io.imsave('Task09_Spleen/labelsTr_png' + label_path[i][22:-7]+ '_' + str(j) + '.png', label2d)
        '''    
        
