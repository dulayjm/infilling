# Potentially just make this script a notebook

import argparse
import csv
import numpy as np
import os
import pandas as pd
import torch
from torchvision import transforms

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('-dataset', default='dataset/train_set.csv', required=True,
                    help='file to operate upon')

args = parser.parse_args()

datafile = args.dataset

with open(datafile) as cf: 
    csv_reader = csv.reader(cf,delimiter=',')
    classifiers = []
    unique_classifiers = []
    imgs = []
    labels = []

    for row in csv_reader:
        classifier = row[1]
        classifiers.append(classifier)
        # if classifier not in unique_classifiers:
        #     unique_classifiers.append(classifier)
        img = row[2]
        imgs.append(img)
        label = row[3]
        labels.append(label)

    dataframe = pd.DataFrame({"classifiers": classifiers, "imgs": imgs, "labels": labels})
    
    """Before loss, we need to set up the data as batches"""
    dataframe.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # $L = max(f||f(A) - f(P)||^2 - f||f(A) - f(P)||^2 + \alpha, 0)$
    train_data = dataframe.sample(frac=0.7).reset_index()

