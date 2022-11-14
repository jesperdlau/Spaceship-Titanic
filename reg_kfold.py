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
from model import RegressionModelHyper1, RegressionModelHyper3, RegressionModel
from nnRegTrain import train, test, train2, test2


# Hyperparameters 
source_path = "Spaceship-Titanic/Data/data_regression.csv"
save_path = "Spaceship-Titanic/Data/nn_reg_pred.npy"
k_folds = 5
#test_size= 0.1
batch_size = 5
batch_size_outer = 1
lr = 1e-4
lr_outer = 1e-4 # Potential different batch size, lr and epoch range for final outer training
epochs = 5
epochs_outer = 20
hyper_range = 3
loss_fn = MSELoss()
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Prepare data
data = pd.read_csv(source_path)
dataset = SpaceshipDataset(data)

# Set kfold. Important to not shuffle so we can reproduce the splits. The data is already shuffeled. 
kfold = KFold(n_splits=k_folds, shuffle=False)

# Performance tracking
best_hyper_list = [0]*5
best_pred_list = [0]*5
best_loss_list = [0]*5

# Outer layer
for outer_layer, (train_id_outer, test_id_outer) in enumerate(kfold.split(data)):

    # Track loss of all hyperparameters for each inner layer. 2d array. (Inner layer x hyper)
    inner_hyper_loss = np.zeros((5, 3))
    
    for inner_layer, (train_id_inner, test_id_inner) in enumerate(kfold.split(train_id_outer)):
        # Dataloaders used for inner layer AND hyper parameter tuning
        dataloader_train_inner = DataLoader(dataset, batch_size=batch_size, sampler=train_id_inner)
        dataloader_test_inner = DataLoader(dataset, batch_size=batch_size, sampler=test_id_inner)

        # Hyper parameter layer
        for hyper in range(hyper_range):
            # Reset model and optimizer
            model = RegressionModelHyper3(hyper=hyper)
            optimizer = Adam(model.parameters(), lr=lr)

            # Training/test loop
            best_hyper_loss = 10
            for epoch in range(epochs):
                train_loss, model_state = train2(dataloader_train_inner, model, loss_fn, optimizer)
                test_loss = test2(dataloader_test_inner, model, loss_fn)

                if test_loss < best_hyper_loss:
                    best_hyper_loss = test_loss
                #print(f"{epoch} loss: {test_loss:.4f}")

            # Save lowest error for each hyper to compute generalization error
            inner_hyper_loss[inner_layer, hyper] = best_hyper_loss 

            # Print current progress
            print(f"[{outer_layer+1}/{k_folds}:{inner_layer+1}/{k_folds}:{hyper+1}/{hyper_range}: {best_hyper_loss:.4f}]")

    # Generalization error as mean of loss for each hyper. 
    gen_error = np.mean(inner_hyper_loss, axis=0)
    
    # Find best model for inner model. The one with lowest mean loss. Save it. 
    best_hyper = np.argmin(gen_error)
    best_hyper_list[outer_layer] = best_hyper

    # Dataload for outer layer
    dataloader_train_outer = DataLoader(dataset, batch_size=batch_size_outer, sampler=train_id_outer)
    dataloader_test_outer = DataLoader(dataset, batch_size=1, sampler=test_id_outer)
    
    # Model is initialized with best hyper 
    model = RegressionModelHyper3(hyper=best_hyper)
    optimizer = Adam(model.parameters(), lr=lr)

    # Outer train/test loop
    best_outer_fold, best_outer_loss = 0, 10
    for epoch in range(epochs_outer):
        train_loss, model_state = train2(dataloader_train_outer, model, loss_fn, optimizer)
        test_loss, pred_arr = test(dataloader_test_outer, model, loss_fn)

        # Save prediction array for best epoch of best model
        if test_loss < best_outer_loss:
            best_outer_fold, best_outer_loss = outer_layer, test_loss
            best_pred_list[outer_layer] = pred_arr
            best_loss_list[outer_layer] = best_outer_loss
        
        # Print
        print(f"Outer layer: {outer_layer} Epoch: {epoch} - loss: {best_outer_loss:.4f}")

    # Print
    print("#"*10)
    print(inner_hyper_loss)
    print(f"Outer layer: {outer_layer} - best hyper: {best_hyper} - loss: {best_outer_loss:.4f}")
    print("####")

print("\n\n")
print("-"*10)
print(f"best_hyper_list:\n{best_hyper_list}")
print(f"best_loss_list:\n{best_loss_list}")
print()

# Save predictions to .npy file
best_pred_arr = np.array(best_pred_list)
np.save(save_path, best_pred_arr)
print("saved predictions to numpy object")


