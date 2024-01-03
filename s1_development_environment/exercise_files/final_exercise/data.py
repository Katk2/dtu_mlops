import torch
import numpy as np
from torchvision import datasets, transforms
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    BASEDIR = 'C:/Programmering/dtu_mlops/data/corruptmnist'
    imgTrain = 'train_images_0.pt'
    labelTrain = 'train_target_0.pt'
    imgTest = 'test_images.pt'
    labelTest = 'test_target.pt'

    

    imgPathTrain = 'C:/Programmering/dtu_mlops/data/corruptmnist/train_images_0.pt' 
    labelPathTrain = 'C:/Programmering/dtu_mlops/data/corruptmnist/train_target_0.pt'
    imgPathTest = 'C:/Programmering/dtu_mlops/data/corruptmnist/test_images.pt'
    labelPathTest = 'C:/Programmering/dtu_mlops/data/corruptmnist/test_target.pt'

    dataTrain = torch.load(imgPathTrain).view(-1, 1, 28, 28)
    labelsTrain = torch.load(labelPathTrain)
    dataTest = torch.load(imgPathTest).view(-1, 1, 28, 28)
    labelsTest = torch.load(labelPathTest)
    

    trainDataset = [[x, y] for x, y in zip(dataTrain, labelsTrain)]
    testDataset = [[x, y] for x, y in zip(dataTest, labelsTest)]


    train = DataLoader(trainDataset, batch_size=64, shuffle=True)
    test = DataLoader(testDataset, batch_size=64, shuffle=True)


    return train, test
