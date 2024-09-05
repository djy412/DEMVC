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
from classes.ResNet18_autoencoder import ResNetAutoencoder
from classes.DEC_Model import ClusterLayer
from scripts.data_loading import get_MVPN_dataloaders, get_Multi_Market_dataloaders
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, v_measure_score
from scipy.optimize import linear_sum_assignment
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import torch.nn.functional as F

Pre_train = True

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
train_dataloader = DataLoader(MNIST_Data, batch_size=BATCH_SIZE, shuffle=True)

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

#*****************************************************************************
#--- Create a plot function for t-SNE
#*****************************************************************************
def plot_projection(x, colors):
    
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(10):
        plt.scatter(x[colors == i, 0],
                    x[colors == i, 1])

    for i in range(10):#--- Put the label in the cluster center
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
#*****************************************************************************



#************************************************************************
#--- Define MSE and KL Loss function
#************************************************************************
class MSE_KL_DEV_Loss(nn.Module):
    def __init__(self, LAMBDA = LAMBDA):
        super(MSE_KL_DEV_Loss, self).__init__()
        self.LAMBDA = LAMBDA
        
    def forward(self, x_hat1, x1, x_hat2, x2, q):
        
        # Calculate the squared differences
        squared_diff = (x_hat1 - x1) ** 2
        # Calculate the mean of the squared differences
        mse1 = torch.mean(squared_diff)
        squared_diff = (x_hat2 - x2) ** 2
        mse2 = torch.mean(squared_diff)
            
        #--- This returns a "sharpened" version of the input distribution q normalized 
        weight = q ** 2 / q.sum(0)
        p = (weight.T / weight.sum(1)).T
        
        #--- Calculate the KL divergence between two distributions.
        # Ensure that p and q are probability distributions (sum to 1)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        kl_div = F.kl_div(q.log(), p, reduction='batchmean') #-- ensure that the output is averaged over the batch
        
        loss = mse1+ mse2 + LAMBDA*kl_div
        
        return loss
#************************************************************************


#*****************************************************************************
#--- Create a function for cluster accuracy
#************************************************************************
def cluster_accuracy(y_true, y_pred):

    count_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for i in range(y_pred.size):
        count_matrix[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_pred.size
    return accuracy
#************************************************************************
NMI = normalized_mutual_info_score
vmeasure = v_measure_score
ARI = adjusted_rand_score


#**************************************************************
#--- Training loop
#**************************************************************
if Pre_train:
    lost_history = []
    AE1.train()
    AE2.train()
    for epoch in range(EPOCHS):
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
        lost_history.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
    #**************************************************************
    
    # #--- Save the wieghts to a file
    torch.save(AE1, 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/weights/AE1_model.pt')
    torch.save(AE2, 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/weights/AE2_model.pt')
    
else:
    #--- Load the weights from the previouly trained models
    load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/weights/AE1_model.pt'    
    AE1 = ResNetAutoencoder(LATENT_DIM).to(device) 
    AE1.load_state_dict(torch.load(load_model_path)) 
    load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/weights/AE2_model.pt'    
    AE2 = ResNetAutoencoder(LATENT_DIM).to(device) 
    AE2.load_state_dict(torch.load(load_model_path)) 

#--- Show Training loss
counter = range(0,len(lost_history))
plt.figure("Loss")
plt.plot(counter, lost_history)    
plt.xlabel(f"Pre-train Loss per Epoch-{EPOCHS}Epochs")          
plt.show 
    
#--- Show the latent variables using true labels t-SNE 
tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(z1.detach().cpu())       
plot_projection(X_tsne, y_true)
plt.title("Pretrained Latent Space")
plt.xlabel(EPOCHS)
plt.show() 
    
# Convert the output to images
output1 = output1.view(output1.size(0), 1, 32, 32).cpu().data

# Plot the original and reconstructed images
fig, axes = plt.subplots(2, 5, figsize=(12, 2))
for i in range(5):
    axes[0, i].imshow(x1[i].cpu().view(32, 32))
    axes[1, i].imshow(output1[i].cpu().view(32, 32))
    axes[0, i].axis('off')
    axes[1, i].axis('off')
plt.show()

#**************************************************************
# Step 1: initialize cluster centers using k-means
#--- Get the z for cluster assignment 
DEC = ClusterLayer(z1, NUM_CLASSES).to(device)

optimizer_C = optim.Adam(DEC.parameters(), lr=LR)
criterion = MSE_KL_DEV_Loss()
#criterion = nn.KLDivLoss(reduction='batchmean')

copy_z1 = z1
copy_z1 = copy_z1.detach().cpu()

#--- k-means
kmeans = KMeans(n_clusters=DEC.n_clusters, n_init=100)
k_perdict = kmeans.fit_predict(copy_z1)
cluster_centers = kmeans.cluster_centers_  
#cluster_centers = torch.tensor(cluster_centers).to(device)
DEC.cluster_centers.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)


#--- See how accurate the inital cluster centers are to the real labels
acc1 = np.round(cluster_accuracy(y_true, k_perdict), 5)
nmi1 = np.round(NMI(y_true, k_perdict), 5)
vmea1 = np.round(vmeasure(y_true, k_perdict), 5)
ari1 = np.round(ARI(y_true, k_perdict), 5)
print('Initial k_means acc: acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f' % (acc1, nmi1, vmea1, ari1))

#**************************************************************
#--- Step 2: deep clustering 
#**************************************************************
lost_history = []
z = torch.stack((z1, z2))

for epoch in range(EPOCHS):
    running_loss = 0.0
    #--- This is the "collaborative" training part  
    for view in range(2):#--- only have two views currently         
        for data in train_dataloader:
            x1, x2, y_true = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            # Forward pass
            output1, z1 = AE1(x1)
            output2, z2 = AE2(x2)
            z = torch.stack((z1, z2))
            
            q = DEC(z[view])      
            
            #--- Get updated loss based on MSE and KL divergence of target disturbutions 
            loss = criterion(output1, x1, output2, x2, q) 
            
            # Backward pass
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            optimizer_C.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            optimizer_C.step()
            
            running_loss += loss.item() * x1.size(0)
        
    epoch_loss = running_loss / len(train_dataloader.dataset)
    lost_history.append(epoch_loss)
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")
 
    #--- Print the Scores of classification from the embeddings 
    #--- Get the most likely cluster predictions per view
    #prediction  = torch.max(z1, dim=1).indices
    #y_pred = np.array(prediction.cpu())
copy_z1 = z1
y_pred = kmeans.fit_predict(copy_z1.detach().cpu())
acc2 = np.round(cluster_accuracy(y_true, y_pred), 5)
nmi2 = np.round(NMI(y_true, y_pred), 5)
vmea2 = np.round(vmeasure(y_true, y_pred), 5)
ari2 = np.round(ARI(y_true, y_pred), 5)
print('z1: acc=%.5f, nmi=%.5f, v-measure=%.5f, ari=%.5f' % (acc2, nmi2, vmea2, ari2))

#--- Show the loss histroy 
counter = range(0,len(lost_history))
plt.figure("Loss")
plt.plot(counter, lost_history)    
plt.xlabel(f"Fine tune Loss per Epoch-{EPOCHS}Epochs")          
plt.show 

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

#--- Show the latent variables using true labels t-SNE 
tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(z1.detach().cpu())       
plot_projection(X_tsne, y_true)
plt.title("Refined Latent Space")
plt.xlabel(EPOCHS)

plt.show() 
acc2, nmi2, vmea2, ari2

plt.text(0.1, 0.8, 'Initial Acc is: %.3f' %(acc1))
plt.text(0.1, 0.7, 'Final Acc is: %.3f' %(acc2))
plt.text(0.1, 0.6, 'Initial NMI is: Acc %.3f' %(nmi1))
plt.text(0.1, 0.5, 'Final NMI is: Acc %.3f' %(nmi2))
plt.text(0.1, 0.3, 'Epochs: %.1f' %(EPOCHS))
plt.text(0.1, 0.1, 'Dataset Fashion')
#plt.text(0.1, 0.0, 'Triplet Loss')
plt.axis('off')
plt.show()



