# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 09:44:57 2024
@author: djy41
"""
from __future__ import division, print_function
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
from torchvision import transforms
import pandas as pd
import os
from skimage.io import imread

#**************************************************************
#--- Create a dataset of MNIST data
#**************************************************************
class MNIST_Dataset(Dataset):
    """Dataset made with MNIST 32 x 32 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        x = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        x2 = imread(img_path)
        y = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            x = self.transform(x)
            x2 = self.transform(x2)

        return x, x2, y, index
#**************************************************************


#**************************************************************
#--- Create a dataset of Eglin data
#**************************************************************
class Multi_Eglin_Dataset(Dataset):
    """Dataset made with Multi-Eglin 224 x 224 images"""          
    def __init__(self, csv_file, root_dir, transform=None):
          self.annotations = pd.read_csv(root_dir+csv_file)
          self.root_dir = root_dir
          self.transform = transform

    def __len__(self):
        return len(self.annotations)  

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        anchor = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        positive = imread(img_path)
        #img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        #negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            #negative = self.transform(negative)
            
        return anchor, positive, y_label, index
#**************************************************************


#######################################################
# Evaluate Critiron
#######################################################
def cluster_acc(y_true, y_pred, NUM_CLUSTERS):
    count_matrix = np.zeros((NUM_CLUSTERS, NUM_CLUSTERS), dtype=np.int64)
    for i in range(y_pred.size):
        count_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_pred.size
    return accuracy



