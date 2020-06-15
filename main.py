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
# Device
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# Num epochs
num_epochs = 25

# Model
model = models.resnet50()
dfnet_model = models.resnet18()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Batch size
batch_size = 32

# Data set
train_path = '/lab/vislab/DATA/CUB/images/'

# Loss function
criterion = losses.TripletMarginLoss(margin=0.1)




# Run Script
print(type(model))
model.to(device)


# perhaps parameterize the train model to take in our pre-processed data

trained_model, loss = train_model()
print(loss)


print(type(trained_model))
trained_model.test()

print("Finished.")


