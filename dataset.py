from cProfile import label
import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.io import read_image
from torchvision.transforms import Resize, Compose

import os
import pandas as pd
import numpy as np


data_path = '/home/korol/workspace/pfid/'


class HorseDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.path = df['path'].values
        self.labels = df['id'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.path[idx]
        img = read_image(path)
        
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).float()

        return img, label

transforms = Compose([
    #ToTensor(),
    Resize([64,64]),
])


def get_dataloaders(dataset):

    train_loader = DataLoader(dataset,
                              batch_size=20,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True, drop_last=True)

    return train_loader


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    df = pd.read_csv('/home/korol/workspace/pfid/data/input.csv')
    
    dataset = HorseDataset(df, transform=transforms)

    print(len(dataset))
    print(dataset[0])

    ### check dataset
    # for i in range(5):
    #     plt.subplot(1,5,i+1)
    #     plt.imshow(dataset[i+1000][0].transpose_(1,2).transpose_(0,2).numpy())
    #     plt.title(f'label - {dataset[i][1]}', fontsize=10)  
    # #plt.suptitle(f'label - {dataset[i][1]}', fontsize=16)
    # #plt.tight_layout()
    # plt.show()

    train_loader = get_dataloaders(dataset)

    #check dataloader
    batch = next(iter(train_loader))

    print(batch[0].shape)
    print(batch[1].shape)

    images = batch[0]
    labels = batch[1]
    plt.figure(figsize=(14,7))
    for i in range(8):
        plt.subplot(1,8,i+1)
        plt.imshow(images[i].transpose_(1,2).transpose_(0,2).numpy())
        plt.title(f'label - {labels[i]}', fontsize=10)
    plt.tight_layout()
    plt.show()





