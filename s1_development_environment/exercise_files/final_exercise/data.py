import torch
import numpy as np
from torchvision import datasets, transforms





def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    datapath = torch.load('data/corrupted_mnist.pt')
    transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])
    trainset = datasets.MNIST(datapath, download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    
    
    
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784)
    return train, test
