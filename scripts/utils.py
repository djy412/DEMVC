"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE

def train_epoch(Encoder, DecoderC, DecoderP, device, dataloader, loss_fn, optimizer1, optimizer2, optimizer4):
    """The training loop of autoencoder"""
    #cae.train()#---Set train mode for both the encoder and the decoder
    Encoder.train()
    DecoderC.train()
    DecoderP.train()
    
    train_loss = []
    for _, (a_batch, p_batch, n_batch, y_batch) in tqdm(enumerate(dataloader), total=int(len(dataloader.dataset)/dataloader.batch_size)): #--- This is for triplet batches      
        #---Move tensor to the proper device
        a_batch = a_batch.to(device)
        p_batch = p_batch.to(device)
        n_batch = n_batch.to(device)
        
        z_a_batchC, z_a_batchP = Encoder(a_batch)
        z_p_batchC, z_p_batchP = Encoder(p_batch)
        z_n_batchC, z_n_batchP = Encoder(n_batch)
   
        decoded_a_batch = DecoderC(z_a_batchC) + DecoderP(z_a_batchP)
        decoded_p_batch = DecoderC(z_p_batchC) + DecoderP(z_p_batchP)
        decoded_n_batch = DecoderC(z_n_batchC) + DecoderP(z_n_batchP)        

        #---Evaluate loss
        loss = loss_fn(a_batch, p_batch, n_batch, decoded_a_batch, decoded_p_batch, decoded_n_batch, z_a_batchC, z_p_batchC, z_n_batchC)

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer4.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer4.step()

        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)
#************************************************************************

def test_epoch(Encoder, DecoderC, DecoderP, device, dataloader, loss_fn):
    """The validation loop of autoencoder on the test dataset"""
    # Set evaluation mode for encoder and decoder
    Encoder.eval()
    DecoderC.eval()
    DecoderP.eval()
    
    with torch.no_grad(): # No need to track the gradients
        #---Define the lists to store the outputs for each batch
        #decoded_data = []
        #original_data = []
        for x_batch, _, _, _ in dataloader: #---This is for Triplet batches
            # Move tensor to the proper device
            x_batch = x_batch.to(device)
            # CAE data
            z1, z2 = Encoder(x_batch) 
            decoded_batch = DecoderC(z1) + DecoderP(z2)
            # Append the network output and the original image to the lists
            decoded_data = decoded_batch.cpu()#.append(decoded_batch.cpu())
            original_data = x_batch.cpu()#.append(x_batch.cpu())
        # Create a single tensor with all the values in the lists
        #decoded_data = torch.cat(decoded_data)
        #original_data = torch.cat(original_data)
        # Evaluate global loss
        val_loss = loss_fn(decoded_data, original_data)

    return val_loss.data, z1, z2
#************************************************************************

def plot_ae_outputs(Encoder, DecoderC, DecoderP, dataset_opt, epoch, dataset, device, n=10):
    """Saving plot diagrams with reconstructed images in comparision with the original ones for a visual assessment"""
    
    plt.figure(figsize=(16,4.5))
    for i in range(n):

        ax = plt.subplot(2,n,i+1)
        img = dataset[i][0]
        labels = dataset[i][3]
        
        plt.imshow(img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Original images from ' + dataset_opt + ' epoch=' + str(epoch))

        ax = plt.subplot(2, n, i + 1 + n)
        img = img.unsqueeze(0).to(device) # img -> (3, xx, xx) but img.unsqueeze(0) -> (1,3,xx,xx)
        #cae.eval()
        Encoder.eval()
        DecoderC.eval()
        DecoderP.eval()
        
        with torch.no_grad():
            z1, z2 = Encoder(img)
            rec_img = DecoderC(z1) + DecoderP(z2)
        rec_img = rec_img.cpu().squeeze() # rec_img -> (1, 3, xx, xx) but img.squeeze() -> (3,xx,xx)
        plt.imshow(rec_img.permute((1, 2, 0))) # rgb
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == n//2:
            ax.set_title('Reconstructed images from ' + dataset_opt + ' epoch=' + str(epoch))

    if not os.path.isdir('output'):
        os.mkdir('output')
    # plt.show()
    plt.savefig(f'output/{epoch}_epoch_from_{dataset_opt}.png')
    
#************************************************************************

def checkpoint(model1, model2, model3, epoch, val_loss, filenameE, filenameDC, filenameDP):
    """Saving the model at a specific state"""
    torch.save(model1.state_dict(), filenameE)
    torch.save(model2.state_dict(), filenameDC)
    torch.save(model3.state_dict(), filenameDP)
    
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model1.state_dict(),
    #         # 'optimizer_state_dict': optimizer.state_dict(),
    #         'val_loss': val_loss,
    #         }, filenameE)
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model2.state_dict(),
    #         # 'optimizer_state_dict': optimizer.state_dict(),
    #         'val_loss': val_loss,
    #         }, filenameDC)
    # torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model3.state_dict(),
    #         # 'optimizer_state_dict': optimizer.state_dict(),
    #         'val_loss': val_loss,
    #         }, filenameDP)
#************************************************************************

def resume(model, filename):
    """Load the trained autoencoder model"""
    checkpoint = torch.load(filename)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['val_loss']
    return model, epoch, loss
#************************************************************************







