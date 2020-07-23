from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
from pytorch_metric_learning import losses, samplers
from random import randint
from skimage import io, transform
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.sampler import Sampler
from torchvision import datasets, models, transforms

# Configurations
# Train
train = True 

# Inpaint
inpaint = False

# Test
test = False

# Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Num epochs
num_epochs = 15

# Model
model = models.resnet50(pretrained=True)
# model = ClassifierSiLU()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Batch size
batch_size = 32

# Data set
train_path = '/lab/vislab/DATA/CUB/images/train/'
# train_path = '/lab/vislab/DATA/just/infilling/samples/places2/mini/'

# Inpainting mask path
mask_path = './samples/places2/mask/'

# Loss function
criterion = losses.TripletMarginLoss(margin=0.05,triplets_per_anchor="all")
# criterion = torch.nn.CosineEmbeddingLoss()

# Mask to use in random masking
mask = Image.open('./samples/places2/mask/mask_01.png')

transformations = transforms.Compose([
    transforms.Resize((256, 256)),
    # RandomMask(mask),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(train_path, transformations)
# test_set = ...

# train_sampler = torch.utils.data.RandomSampler(train_set)
train_sampler = samplers.MPerClassSampler(dataset.targets, 2, len(dataset))

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler, num_workers=4, drop_last=True)

# test_loader = ...


# Parameterized struct to send project settings
config = {
    'train' : train, 
    'inpaint' : inpaint,
    'test' : test,
    'device' : device,
    'mask_path' : mask_path,
    'num_epochs' : num_epochs,
    'model' : model,
    'optimizer' : optimizer,
    'batch_size' : batch_size,
    'train_loader' : train_loader,
}

class Config: 
    def __init__(self):
        self.config = config