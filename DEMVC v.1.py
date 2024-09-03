# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 09:04:58 2024

@author: djy41
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
import math
from scripts.config import BATCH_SIZE, EPOCHS, MODEL_FILENAME1, MODEL_FILENAME2, MODEL_FILENAME3, EARLY_STOP_THRESH, LR, WORKERS, LATENT_DIM, NUM_CLASSES, LAMBDA, CLASSIFIER_EPOCHS, dataset_name
from classes.ResNet18_encoder import ResNet18EncoderFC
from classes.ResNet18_decoder import ResNet18DecoderFC
from classes.ResNet18_autoencoder import ResNetAutoencoder
from scripts.data_loading import get_MVPN_dataloaders, get_Multi_Market_dataloaders
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
from scipy.optimize import linear_sum_assignment

path_to_data = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi Fashion/'
#path_to_data = 'C:/Users/djy41/Desktop/PhD/Code/TripletLoss-Multi View/MNIST Test/'

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#--- Initialize the model, loss function, and optimizer
print("Defining model...")
AE1 = ResNetAutoencoder(LATENT_DIM).to(device)
AE2 = ResNetAutoencoder(LATENT_DIM).to(device)
criterion = nn.MSELoss()
optimizer1 = optim.Adam(AE1.parameters(), lr=1e-3)
optimizer2 = optim.Adam(AE2.parameters(), lr=1e-3)

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
        view1 = imread(img_path)
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        view2 = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        return view1, view2, y_label
#**************************************************************

#--- Load the MNIST Dataset
"""MNIST dataloader with (32, 32) images."""
MNIST_Data = MNIST_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.ToTensor())

#--- Use dataloader to get a batch of images
train_dataloader = DataLoader(MNIST_Data, batch_size=64, shuffle=True)

#--- Show one image
view1, view2s, train_labels = next(iter(train_dataloader))
img = view1[0].squeeze()#.permute(1,2,0)
label = train_labels[0]
plt.imshow(img)
plt.show()
print(f"Feature batch shape: {view1.size()}")
print(f"Labels batch shape: {train_labels.size()}")
print(f"Label: {label}")
size = img.shape
size1 = size[0]
size2 = size[1]
print(f"Image is {size1}x{size2}")
print("Learning Rate ", LR)
print("Epochs are :",EPOCHS)
print("LAMDA1 ",LAMBDA)
print("Data Max value is: ",torch.max(view1))
print("Data Min value is: ",torch.min(view1))

#**************************************************************
#--- Training loop
#**************************************************************
for epoch in range(EPOCHS):
    AE1.train()
    AE2.train()
    running_loss = 0.0
    for data in train_dataloader:
        x1, x2, y_true = data
        x1 = x1.to(device)
        x2 = x2.to(device)
        
        # Forward pass
        output1, z1 = AE1(x1)
        loss1 = criterion(output1, x1)  # Compare the output with the input
        output2, z2 = AE2(x2)
        loss2 = criterion(output2, x2)   
        
        # Backward pass
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        
        running_loss += loss1.item() * x1.size(0)
    
    epoch_loss = running_loss / len(train_dataloader.dataset)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
#**************************************************************

# Step 1: initialize cluster centers using k-means
#--- Get the z for cluster assignment 
z1 = z1.detach().cpu()
z2 = z2.detach().cpu()

#--- k-means
kmeans = KMeans(n_clusters=NUM_CLASSES, n_init=100)
y1 = kmeans.fit_predict(z1)
cluster_centers1 = kmeans.cluster_centers_

y2 = kmeans.fit_predict(z2)
cluster_centers2 = kmeans.cluster_centers_

y_pred = np.array([y1, y2])
print('y_pred', y_pred)

#--- Run testing on Results
NMI = normalized_mutual_info_score
vmeasure = v_measure_score
ARI = adjusted_rand_score

def cluster_accuracy(y_true, y_pred):
    # Convert to numpy if tensors are given
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Create a confusion matrix
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1

    # Solve the linear assignment problem (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Calculate accuracy
    accuracy = sum([w[i, j] for i, j in zip(row_ind, col_ind)]) / len(y_true)
    return accuracy

for i in range(len(y_pred)):
    acc = np.round(cluster_accuracy(y_true, y_pred[i]), 5)
    nmi = np.round(NMI(y_true, y_pred[i]), 5)
    vmea = np.round(vmeasure(y_true, y_pred[i]), 5)
    ari = np.round(ARI(y_true, y_pred[i]), 5)
    print('Start-'+str(y_pred[i])+': acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f' % (acc, nmi, vmea, ari))


#--- Step 2: deep clustering
#--- get the q distribution and p distribution 
def reference_distribution(z, u): #--- Equation 6 or t-distribution
    # Compute the squared difference between each pair of elements
    diff_squared = np.abs(z[:, np.newaxis] - u[np.newaxis, :]) ** 2
    
    # Compute the inverse of (1 + diff_squared)
    inv_distances = (1 + diff_squared) ** -1
    
    # Sum over the second axis (i.e., sum over j)
    sum_inv_distances = inv_distances.sum(axis=1, keepdims=True)
    
    # Calculate q_(i,j) by normalizing with the sum
    q_matrix = inv_distances / sum_inv_distances
    
    return q_matrix # This matrix is 3 dimension per each view (sample#, distance to first cluster center, ..., distance to last cluster center)

def target_distribution(q): #--- Equation 7
    #--- This returns a "sharpened" version of the input distribution q normalized 
    weight = q ** 2 / q.sum(0)
    p = (weight.T / weight.sum(1)).T
    return p

soft_labels_q = reference_distribution(z1, cluster_centers1)

#--- Get the prediction of the i'th sample cluster
q_1 = torch.max(soft_labels_q, dim=1).values
p_1 = target_distribution(q_1)
soft_labels_q = reference_distribution(z2, cluster_centers2)

#--- Get the prediction of the i'th sample cluster
q_2 = torch.max(soft_labels_q, dim=1).values
p_2 = target_distribution(q_1)

def kl_divergence(p, q):
#--- Calculate the KL divergence between two distributions.
    p = np.asarray(p, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    
    # Avoid division by zero or log(0) by adding a small epsilon
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    
    return np.sum(p * np.log(p / q))

print('kl_divergence', kl_divergence(p_1, q_1))





#--- Show the results of training
for data in train_dataloader:
    images, _, _ = data
    images = images.to(device)
    break
    
# Pass through the autoencoder
AE1.eval()
with torch.no_grad():
    output, z = AE1(images)

# Convert the output to images
output = output.view(output.size(0), 1, 32, 32).cpu().data

# Plot the original and reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(12, 2))
for i in range(5):
    axes[0, i].imshow(images[i].cpu().view(32, 32))
    axes[1, i].imshow(output[i].view(32, 32))
    axes[0, i].axis('off')
    axes[1, i].axis('off')
plt.show()







