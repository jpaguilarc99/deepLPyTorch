"""
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet

complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html

On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale

On Tensors
----------
LinearTransformation, Normalize, RandomErasing

Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage

Generic
-------
Use Lambda 

Custom
------
Write own class

Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])
"""
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset

class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0] # filas del wine dataset

        self.x = xy[:, 1:] # todas las filas, y todas las columnas a excepcion del target
        self.y = xy[:, [0]] # todas las filas y unicamente la columna target

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)  
        return sample       
    
    def __len__(self):
        return self.n_samples

# Custom class to convert numpy arrays to torch arrays   
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

# Custom class to multiply by any factor the inputs labels
class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

# we pass totensor class as argument to realize data transformations
dataset = WineDataset(transform=MulTransform(factor=3))
first_data = dataset[0]
features, labels = first_data

# combine the two custom classes to transform the data
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(factor=3)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels), first_data)