# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 22:22:45 2024

@author: djy41
"""
import torch.nn as nn
import torch
from torchvision import models

class ResNetAutoencoder(nn.Module):
    def __init__(self, latent_dim, initial_shape=(512, 1, 1)):
        super(ResNetAutoencoder, self).__init__()
        #--- Load ResNet18 model
        resnet = models.resnet18(weights=False)
        #--- Modify the first convolutional layer to accept a single-channel input
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),  # Changed in_channels to 1
            *list(resnet.children())[1:-2]  # Exclude the first conv layer and the fully connected layers
        )
        
        self.fc1 = nn.Linear(512, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 512)
        self.initial_shape = initial_shape
        #self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # Exclude the last two layers

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        z = self.fc1(x)
        x = self.fc2(z)
        x = x.view(-1, *self.initial_shape)
        y = self.decoder(x)
        
        return y, z
#*****************************************************************************

