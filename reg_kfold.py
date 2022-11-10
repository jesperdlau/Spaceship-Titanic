# Import libraries
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold


# Import from other files
from data_loader import SpaceshipDataset, SpaceData
from model import RegressionModelHyper1
from nnRegTrain import train, test, train2, test2
#from nn-reg-train import 

# Hyperparameters 
source_path = "Spaceship-Titanic/Data/data_regression.csv"
k_folds = 5
#test_size= 0.1
batch_size = 5
lr = 1e-4
epochs = 5
hyper_range = 3
loss_fn = MSELoss()
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Need to reset for every training loop
model_ref0 = RegressionModelHyper1(hyper=0)
model_ref1 = RegressionModelHyper1(hyper=1)
model_ref2 = RegressionModelHyper1(hyper=2)

def opt(hyper=0):
    if hyper == 0: return Adam(model_ref0.parameters(), lr=lr)
    if hyper == 1: return Adam(model_ref1.parameters(), lr=lr)
    if hyper == 2: return Adam(model_ref2.parameters(), lr=lr)


# Prepare data
data = pd.read_csv(source_path)
dataset = SpaceshipDataset(data)


# Set kfold. Important to not shuffle so we can reproduce the splits. The data is already shuffeled. 
kfold = KFold(n_splits=k_folds, shuffle=False)


# Performance tracking
best_outer_fold, best_outer_loss = 0, 10
outer_layer_models = []
best_hyper_list = [0]*5

# Outer layer
for outer_layer, (train_id_outer, test_id_outer) in enumerate(kfold.split(data)):
    

    # Inner layer
    best_inner_fold, best_inner_loss = 0, 10
    best_inner_train_id, best_inner_test_id = 0, 0
    for inner_layer, (train_id_inner, test_id_inner) in enumerate(kfold.split(train_id_outer)):
        # Dataloaders used for inner layer AND hyper parameter tuning
        dataloader_train_inner = DataLoader(dataset, batch_size=batch_size, sampler=train_id_inner)
        dataloader_test_inner = DataLoader(dataset, batch_size=batch_size, sampler=test_id_inner)

        # Hyper parameter layer
        best_hyper, best_hyper_loss = 0, 10
        for hyper in range(hyper_range):
            # Reset model and optimizer
            model = RegressionModelHyper1(hyper=hyper)
            optimizer = opt(hyper)

            # Training/test loop
            for epoch in range(epochs):
                model_state = train2(dataloader_train_inner, model, loss_fn, optimizer)
                test_loss = test2(dataloader_test_inner, model, loss_fn)

                if test_loss < best_hyper_loss:
                    best_hyper, best_hyper_loss = hyper, model_state, test_loss

            



            # Print current progress
            print(f"[{outer_layer+1}/{k_folds}:{inner_layer+1}/{k_folds}:{hyper+1}/{hyper_range}]")

        # TODO:  Calculate generalization error???


        # Inner layer train/test with best hyperparameter
        # Reset model and optimizer
        # model = RegressionModelHyper1(hyper=best_hyper)
        # optimizer = opt(best_hyper)
        # for epoch in range(epochs):
        #     model_state = train2(dataloader_train_inner, model, loss_fn, optimizer)
        #     test_loss = test2(dataloader_test_inner, model, loss_fn)

        #     if test_loss < best_inner_loss:
        #         best_inner_fold, best_inner_loss = inner_layer, test_loss
        #         best_inner_train_id, best_inner_test_id = train_id_inner, test_id_inner
    
    # Prepare outer loop
    dataloader_train_outer = DataLoader(dataset, batch_size=batch_size, sampler=train_id_outer)
    dataloader_test_outer = DataLoader(dataset, batch_size=batch_size, sampler=test_id_outer)
    model = RegressionModelHyper1(hyper=best_hyper)
    optimizer = opt(best_hyper)

    # Outer train/test loop
    for epoch in range(epochs):
        model_state = train2(dataloader_train_outer, model, loss_fn, optimizer)
        test_loss = test2(dataloader_test_outer, model, loss_fn)

        if test_loss < best_outer_loss:
            best_outer_fold, best_outer_loss = outer_layer, test_loss
        

    print("Done")

    




