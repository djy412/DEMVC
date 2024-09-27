# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 00:02:48 2024
@author: djy41
"""
from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader

from utils import cluster_acc, Multi_Eglin_Dataset, MNIST_Dataset
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from classes.ResNet18_autoencoder import ResNetAutoencoder
from torchvision import transforms

PreTRAIN = True
EPOCHS = 1000
EPOCHS_PreTrain = 500
UPDATE_INTERVAL = 5

BATCH_SIZE = 64

NUM_CLUSTERS = 10
LATENT_DIM = NUM_CLUSTERS

CHANNELS = 3
LR = 0.001

TOLERANCE = 0.001

GAMMA = 0.1
New_size = (128,128)

#path_to_data = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi Fashion/'
#path_to_data = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi MNIST Test/'
path_to_data = 'C:/Users/djy41/Desktop/PhD/Datasets/Cars_On_Runway/Eglin Train/'


#----------------------------------------------------------------------------
class CAE(nn.Module):    
    def __init__(self):
        super(CAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(CHANNELS, 32, 3, padding=1),  # (batch_size, 32, 128, 128)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 32, 64, 64)
            nn.Conv2d(32, 64, 3, padding=1),  # (batch_size, 64, 64, 64)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 64, 32, 32)
            nn.Conv2d(64, 128, 3, padding=1),  # (batch_size, 128, 32, 32)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 128, 16, 16)
            nn.Conv2d(128, 256, 3, padding=1),  # (batch_size, 256, 16, 16)
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # (batch_size, 256, 8, 8)
            nn.Flatten(),
            nn.Linear(256*8*8, LATENT_DIM)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 256*8*8),
            nn.Unflatten(1, (256,8,8)),
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 128, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 64, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 32, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, CHANNELS, 3, stride=2, padding=1, output_padding=1),  # (batch_size, 3, 128, 128)
            #nn.Sigmoid()
            nn.Tanh()
        )
        
    def forward(self, x):
        z = self.encoder(x)
        x_bar = self.decoder(z)
        return x_bar, z
    
#----------------------------------------------------------------------------
class IDEC(nn.Module):
    def __init__(self, n_z, n_clusters, alpha=1):
        
        super(IDEC, self).__init__()
        self.alpha = 1.0
        #self.ae = ResNetAutoencoder(latent_dim = NUM_CLUSTERS)
        self.ae = CAE().to(device)
        
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, path='', view=1):
        if path == '':
            pretrain_ae(self.ae, view)


    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2))
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, z, q

#----------------------------------------------------------------------------
def target_distribution(q):
    weight = q**2 / q.sum(0)
    weight = (weight.t() / weight.sum(1)).t()
    return weight

#----------------------------------------------------------------------------
def pretrain_ae(model, view):

    if PreTRAIN == True:
        '''' Load the Data '''
        Training_Data = Multi_Eglin_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(),]))
        #Training_Data = MNIST_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(New_size),]))
        train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, shuffle=True)
    
        optimizer = Adam(model.parameters(), lr=LR)
        
        for epoch in range(EPOCHS_PreTrain):
            total_loss = 0.0
            if view == 1: #--- Get the first view
                for batch_idx, (x, _, y_true, _) in enumerate(train_loader):
                    x = x.to(device)
        
                    optimizer.zero_grad()
                    x_bar, z = model(x)
                    loss = F.mse_loss(x_bar, x)
                    
                    total_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))
                
            if view == 2: #--- Get the second view
              for batch_idx, (_, x, y_true, _) in enumerate(train_loader):
                  x = x.to(device)
      
                  optimizer.zero_grad()
                  x_bar, z = model(x)
                  loss = F.mse_loss(x_bar, x)
                  
                  total_loss += loss.item()
                  loss.backward()
                  optimizer.step()
      
              print("epoch {} loss={:.4f}".format(epoch, total_loss / (batch_idx + 1)))      
        if view == 1:
            torch.save(model.state_dict(), 'data/ae_pretrain1.pkl')
            print("model saved to data/ae_pretrain1.pkl")
        if view == 2:
            torch.save(model.state_dict(), 'data/ae_pretrain2.pkl')
            print("model saved to data/ae_pretrain2.pkl")
        
    else:
        
        if view == 1:
            #load_model_path = '/home/IHMC/dyates/pytorch/DEMVC/data/ae_pretrain1000.pkl'    
            load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/data/ae_pretrain1.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        if view == 2:
            #load_model_path = '/home/IHMC/dyates/pytorch/DEMVC/data/ae_pretrain1000.pkl'    
            load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/data/ae_pretrain2.pkl'    
            model.load_state_dict(torch.load(load_model_path)) 
        
        '''' Load the Data '''
        #Training_Data = Multi_Eglin_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(),]))
        Training_Data = MNIST_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(New_size),]))
        train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, shuffle=True)
        
        for batch_idx, (x, _, _, _) in enumerate(train_loader):
            x = x.to(device)
            x_bar, z = model(x)
            
    #--- Plot the original and reconstructed images
    #output1 = x_bar.view(x_bar.size(0), 1, 32, 32).cpu().data
    fig, axes = plt.subplots(2, 4, figsize=(12, 2))
    for i in range(4):
        if CHANNELS == 3:
            axes[0, i].imshow(x[i].cpu().squeeze().permute(1,2,0))
            axes[1, i].imshow(x_bar[i].detach().cpu().squeeze().permute(1,2,0))
        else:
            axes[0, i].imshow(x[i].cpu().squeeze())
            axes[1, i].imshow(x_bar[i].detach().cpu().squeeze())
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()
    
    #--- Clear everything in memory now that the models are saved and displayed
    x = None
    x_bar = None
    z = None
    torch.cuda.empty_cache()
    
#*****************************************************************************
#--- Create a plot function for t-SNE
#*****************************************************************************
def plot_projection(x, colors):

    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(NUM_CLUSTERS):
        plt.scatter(x[colors == i, 0], x[colors == i, 1])

    for i in range(NUM_CLUSTERS):#--- Put the label in the cluster center
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=16)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
#*****************************************************************************


#*****************************************************************************
def train_DEMVC():
    model = IDEC(n_z = LATENT_DIM, n_clusters=NUM_CLUSTERS, alpha=1.0).to(device)
    model2 = IDEC(n_z = LATENT_DIM, n_clusters=NUM_CLUSTERS, alpha=1.0).to(device)

    model.pretrain(view=1)
    model2.pretrain(view=2)

    '''' Load the Data '''
    Training_Data = Multi_Eglin_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(),]))
    #Training_Data = MNIST_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(New_size),]))
    train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, shuffle=False)
    
    optimizer = Adam(model.parameters(), lr=LR)
    optimizer2 = Adam(model2.parameters(), lr=LR)
    
    #--- Initialize cluster centers 
    data=[]
    y_true = []
    for x, _, y, _ in train_loader:
        data.append(x)
        y_true.append(y)
    data = np.concatenate(data)
    y_true = np.concatenate(y_true)
    data = torch.Tensor(data).to(device)

    x_bar, z = model.ae(data)
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y_true)
    print("model nmi score={:.4f}".format(nmi_k))
    
    z = None
    x_bar = None
    
    #--- Load the initial cluster centers into model
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    #--- Both models share the first views cluster centers per the paper
    model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) 
    
    #--- Get second view cluster centers
    # x_bar, z = model2.ae(data)
    # kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # nmi_k = nmi_score(y_pred, y_true)
    # print("model nmi score={:.4f}".format(nmi_k))
    
    # z = None
    # x_bar = None
    #     #--- Load the initial cluster centers into model
    # model2.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)    
    
    y_pred_last = y_pred
    nmi_view1 = []
    nmi_view2 = []
    Ref_view = 1
    
    #--- Train DEMVC
    for epoch in range(EPOCHS):
        x_bar2 = None
        z2 = None
        q2 = None
        z = None
        q = None
        
        #--- Calculate P of view1
        if Ref_view == 1: 
            if epoch % UPDATE_INTERVAL == 0:    
                tmp_q = None
                z = None
                p = None
                x = None
                print("View1 Reference")
                x, z, tmp_q = model(data)
                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)
        
                # evaluate clustering performance
                y_pred = tmp_q.cpu().numpy().argmax(1)
                
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
        
                acc = cluster_acc(y_true, y_pred, NUM_CLUSTERS)
                nmi = nmi_score(y_true, y_pred)
                ari = ari_score(y_true, y_pred)
                print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                      ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
                nmi_view1.append(nmi)
                
                if epoch > 0 and delta_label < TOLERANCE:
                    print('delta_label {:.5f}'.format(delta_label), '< tol', TOLERANCE)
                    print('Reached tolerance threshold. Stopping training.')
                    break
                Ref_view = 2
                
        #--- Calculate P of view2        
        elif Ref_view == 2: 
            if epoch % UPDATE_INTERVAL == 0:   
                tmp_q = None
                z = None
                p = None
                x = None
                print("View2 Reference")
                x, z, tmp_q = model2(data)
                # update target distribution p
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)
        
                # evaluate clustering performance
                y_pred = tmp_q.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = y_pred
        
                acc = cluster_acc(y_true, y_pred, NUM_CLUSTERS)
                nmi = nmi_score(y_true, y_pred)
                ari = ari_score(y_true, y_pred)
                print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                      ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))
                nmi_view2.append(nmi)
                
                #--- Show the latent variables using true labels t-SNE 
                # tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000)
                # X_tsne = tsne.fit_transform(z.detach().cpu())       
                # plot_projection(X_tsne, y_true)
                # plt.title("Refined Latent Space")
                # plt.xlabel(EPOCHS)
                
                if epoch > 0 and delta_label < TOLERANCE:
                    print('delta_label {:.5f}'.format(delta_label), '< tol', TOLERANCE)
                    print('Reached tolerance threshold. Stopping training.')
                    break
                Ref_view = 1
            
        for batch_idx, (x, x2, _, idx) in enumerate(train_loader):
            x = x.to(device)
            x2 = x2.to(device)
            idx = idx.to(device)
              
            x_bar, z, q = model(x)
                   
            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx], reduction='batchmean')
            loss = GAMMA * kl_loss + reconstr_loss
            
            x_bar2, z2, q2 = model2(x2)
            
            reconstr_loss2 = F.mse_loss(x_bar2, x2)
            kl_loss2 = F.kl_div(q2.log(), p[idx], reduction='batchmean')
            loss2 = GAMMA * kl_loss2 + reconstr_loss2
             
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

        

    #---Save the models
    torch.save(model.state_dict(), 'IDEC_Model1.pt')
    print("model saved as {}.".format('IDEC_Model1.pt'))
    torch.save(model2.state_dict(), 'IDEC_Model2.pt')
    print("model saved as {}.".format('IDEC_Model2.pt'))
 
    #--- Plot the NMI of each view over time
    axis = range(len(nmi_view1))
    plt.figure("V1_NMI")
    plt.title("NMI over training")
    plt.plot(axis, nmi_view1, label = 'V1')
    plt.plot(axis, nmi_view2, label = 'V2')
    plt.xlabel(axis)
    plt.show
    
    #--- Plot the original and reconstructed images
    fig, axes = plt.subplots(2, 4, figsize=(12, 2))
    for i in range(4):
        if CHANNELS == 3:        
            axes[0, i].imshow(x[i].detach().cpu().squeeze().permute(1,2,0) )
            axes[1, i].imshow(x_bar[i].detach().cpu().squeeze().permute(1,2,0) )
        else:
            axes[0, i].imshow(x[i].detach().cpu().squeeze())
            axes[1, i].imshow(x_bar[i].detach().cpu().squeeze())
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    plt.show()
    
#----------------------------------------------------------------------------
if __name__ == "__main__":
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_DEMVC() 

    #--- Clear memory now that the models are trained and saved
    torch.cuda.empty_cache()

    '''' Load the Data '''
    Training_Data = Multi_Eglin_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(),]))
    #Training_Data = MNIST_Dataset(csv_file = 'data.csv', root_dir = path_to_data, transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(New_size),]))
    train_loader = DataLoader(Training_Data, batch_size=BATCH_SIZE, shuffle=True)
    
    ''' Load the saved Models'''
    #load_model_path = '/home/IHMC/dyates/pytorch/DEMVC/IDEC_Model.pt'                
    load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/IDEC_Model1.pt'    
    model1 = IDEC( n_z=LATENT_DIM, n_clusters=NUM_CLUSTERS, alpha=1.0).to(device)
    model1.load_state_dict(torch.load(load_model_path)) 
    model1.eval()
    load_model_path = 'C:/Users/djy41/Desktop/PhD/Code/DEMVC/IDEC_Model2.pt'    
    model2 = IDEC( n_z=LATENT_DIM, n_clusters=NUM_CLUSTERS, alpha=1.0).to(device)
    model2.load_state_dict(torch.load(load_model_path)) 
    model2.eval()
    
    #--- Run through all the data again and display the embedded space
    data=[]
    y_true = []
    for x, _, y, _ in train_loader:
        data.append(x)
        y_true.append(y)
    data = np.concatenate(data)
    y_true = np.concatenate(y_true)
    data = torch.Tensor(data).to(device)

    _, _, q1 = model1(data)
    _, z, q2 = model2(data)
    
    #--- Average the predictions of all views to make the final prediction
    q = (q1 + q2) / 2
    
    # evaluate clustering performance
    y_pred = q.detach().cpu().numpy().argmax(1)
    acc = cluster_acc(y_true, y_pred, NUM_CLUSTERS)
    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    
    #--- Show the latent variables using true labels t-SNE 
    tsne = TSNE(n_components=2, verbose=0, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(z.detach().cpu())       
    plot_projection(X_tsne, y_true)
    plt.title("Refined Latent Space")
    plt.xlabel(EPOCHS)
    
    plt.figure("Scores")
    plt.figure(figsize=(4, 4))
    #plt.text(0.1, 0.8, 'Initial Acc is: %.3f' %(acc))
    plt.text(0.1, 0.7, 'Final Acc is: %.3f' %(acc))
    #plt.text(0.1, 0.6, 'Initial NMI is: Acc %.3f' %(nmi1))
    plt.text(0.1, 0.5, 'Final NMI is: Acc %.3f' %(nmi))
    plt.text(0.1, 0.3, 'Epochs: %.1f' %(EPOCHS))
    plt.text(0.1, 0.1, 'Dataset Multi-Eglin')
    #plt.text(0.1, 0.0, 'Triplet Loss')
    plt.axis('off')
    plt.show()
    plt.savefig("Scores.jpg")
    
    