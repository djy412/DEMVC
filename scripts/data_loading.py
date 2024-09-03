from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from skimage.io import imread
import torch

def get_resized_transform():
    transform = transforms.Compose([
      transforms.Resize((64, 64)),
      transforms.ToTensor()
      # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform
#*****************************************************************************

def get_normalized_transform():
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
    return transform
#*****************************************************************************

def get_simple_transform():
    transform = transforms.Compose([
      transforms.ToTensor()
  ])
    return transform
#*****************************************************************************

def get_cifar10_data_loaders(download, shuffle=False, batch_size=256, num_workers=10):
    # Define the transforms
    # transform = get_resized_transform()
    transform = get_simple_transform()
    if not os.path.isdir('./data'):
        os.mkdir('./data')
    train_dataset = datasets.CIFAR10('./data', train=True, download=download,
                                    transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)

    test_dataset = datasets.CIFAR10('./data', train=False, download=download, 
                                    transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#*****************************************************************************
#--- Loading the MVP-N dataset    
#***************************************************************************** 
def get_MVPN_dataloaders(batch_size=256, num_workers=8):
    """MVP-N dataloader with (224x224) images."""
    train_dataset = MVPN_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/MVP-N_Triplet_Train/', 
                              transform = transforms.ToTensor())
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = MVPN_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/MVP-N_Triplet_Test/', 
                              transform = transforms.ToTensor())
    
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************

#**************************************************************
#--- Create a dataset of MVP-N data
#**************************************************************
class MVPN_Dataset(Dataset):
    """Dataset made with MVP-N 224 x 224 images"""          
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
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        target = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, target
#**************************************************************


#*****************************************************************************
#--- Loading the Multi_Market dataset    
#***************************************************************************** 
def get_Multi_Market_dataloaders(batch_size=256, num_workers=8):
    """Multi_Market dataloader with (64x, 128) images."""
    train_dataset = Multi_Market_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-Market_Triplet_Train/', 
                              transform = transforms.ToTensor())
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=False, shuffle=True)
    
    test_dataset = Multi_Market_Dataset(csv_file = 'data.csv', root_dir = 'C:/Users/djy41/Desktop/PhD/Datasets/Multi-Market_Triplet_Test/', 
                              transform = transforms.ToTensor())
    
    test_loader = DataLoader(test_dataset, batch_size=2*batch_size, num_workers=num_workers, drop_last=False, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset
#*****************************************************************************



#**************************************************************
#--- Create a dataset of Multi_Market data
#**************************************************************
class Multi_Market_Dataset(Dataset):
    """Dataset made with Multi-Market 64 x 128 images"""          
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
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 2])
        negative = imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 3]))
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative, y_label
#**************************************************************


if __name__=='__main__':
    # Load cifar-10 data
    BATCH_SIZE = 32
    print('Downloading & loading data...')
    train_loader, test_loader, train_dataset, test_dataset = get_cifar10_data_loaders(download=True, batch_size=BATCH_SIZE)
    # Check data shapes
    for x_batch, y_batch in train_loader:
        print(x_batch.shape)
        print(y_batch.shape)
        break