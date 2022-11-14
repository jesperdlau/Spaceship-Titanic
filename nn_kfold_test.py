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
#from nn-reg-train import 

# Hyperparameters 
source_path = "Spaceship-Titanic/Data/data_regression.csv"
k_folds = 5
#test_size= 0.1
batch_size = 5
batch_size_outer = 2
lr = 1e-4
lr_outer = 1e-4 # Potential different batch size, lr and epoch range for final outer training
epochs = 5
epochs_outer = 5
hyper_range = 3
loss_fn = MSELoss()
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Need to reset for every training loop
model_ref0 = RegressionModelHyper3(hyper=0)
model_ref1 = RegressionModelHyper3(hyper=1)
model_ref2 = RegressionModelHyper3(hyper=2)


def opt(hyper, learningrate):
    if hyper == 0: return Adam(model_ref0.parameters(), lr=learningrate)
    if hyper == 1: return Adam(model_ref1.parameters(), lr=learningrate)
    if hyper == 2: return Adam(model_ref2.parameters(), lr=learningrate)

#Adam(model_ref2.parameters(), lr=lr)
# Prepare data
data = pd.read_csv(source_path)
dataset = SpaceshipDataset(data)


# Set kfold. Important to not shuffle so we can reproduce the splits. The data is already shuffeled. 
kfold = KFold(n_splits=k_folds, shuffle=False)

#Reset model and optimizer
#model = RegressionModelHyper3(hyper=hyper)
model = RegressionModel()
#optimizer = opt(hyper, lr)
optimizer = Adam(model_ref2.parameters(), lr=lr)
#print(model)

# dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_id_inner)
# dataloader_test = DataLoader(dataset, batch_size=batch_size, sampler=test_id_inner)
dataloader_train = DataLoader(dataset, batch_size=batch_size)
dataloader_test = DataLoader(dataset, batch_size=batch_size)

best_hyper_loss = 10
# Training/test loop
for epoch in range(epochs):
    model_state = train2(dataloader_train, model, loss_fn, optimizer)
    test_loss = test2(dataloader_test, model, loss_fn)
    # epoch_avg_loss, modelstate = train(dataloader_train_inner, model, loss_fn, optimizer)
    # test_loss, pred_arr = test(dataloader_test_inner, model, loss_fn)

    if test_loss < best_hyper_loss:
        best_hyper_loss = test_loss

    print(f"Epoch: {epoch} - loss: {test_loss}")

# # Performance tracking
# best_hyper_list = [0]*5
# best_predictions = [0]*5
# best_loss_list = [0]*5

# # Outer layer
# for outer_layer, (train_id_outer, test_id_outer) in enumerate(kfold.split(data)):
    

#     # Inner layer
#     #best_inner_fold, best_inner_loss = 0, 10
#     #best_inner_train_id, best_inner_test_id = 0, 0

#     # Track loss of all hyperparameters for each inner layer. 2d array. (Inner layer x hyper)
#     inner_hyper_loss = np.zeros((5, 3))
    

#     for inner_layer, (train_id_inner, test_id_inner) in enumerate(kfold.split(train_id_outer)):
#         # Dataloaders used for inner layer AND hyper parameter tuning
#         dataloader_train_inner = DataLoader(dataset, batch_size=batch_size, sampler=train_id_inner)
#         dataloader_test_inner = DataLoader(dataset, batch_size=batch_size, sampler=test_id_inner)

#         # Hyper parameter layer
        
#         for hyper in range(hyper_range):
#             # Reset model and optimizer
#             #model = RegressionModelHyper3(hyper=hyper)
#             model = RegressionModel()
#             #optimizer = opt(hyper, lr)
#             optimizer = Adam(model_ref2.parameters(), lr=lr)
#             #print(model)

#             best_hyper_loss = 10
#             # Training/test loop
#             for epoch in range(epochs):
#                 model_state = train2(dataloader_train_inner, model, loss_fn, optimizer)
#                 test_loss = test2(dataloader_test_inner, model, loss_fn)
#                 # epoch_avg_loss, modelstate = train(dataloader_train_inner, model, loss_fn, optimizer)
#                 # test_loss, pred_arr = test(dataloader_test_inner, model, loss_fn)

#                 if test_loss < best_hyper_loss:
#                     best_hyper_loss = test_loss

#                 print(f"Epoch: {epoch} - loss: {test_loss}")

#             # Save lowest error for each hyper to compute generalization error
#             inner_hyper_loss[inner_layer, hyper] = best_hyper_loss
            

#             # Print current progress
#             print(f"[{outer_layer+1}/{k_folds}:{inner_layer+1}/{k_folds}:{hyper+1}/{hyper_range}]")

#     # Calculate generalization error
#     # Generalization error as mean of loss for each hyper. 
#     gen_error = np.mean(inner_hyper_loss, axis=0)
    
#     # Find best model for inner model. The one with lowest mean loss. Save it. 
#     best_hyper = np.argmin(gen_error)
#     best_hyper_list[outer_layer] = best_hyper

#     # This best hyper model is now trained on outer layer. 
#     # Prepare outer loop
#     dataloader_train_outer = DataLoader(dataset, batch_size=batch_size_outer, sampler=train_id_outer)
#     dataloader_test_outer = DataLoader(dataset, batch_size=1, sampler=test_id_outer)
#     # Model is initialized with best hyper 
#     model = RegressionModelHyper3(hyper=best_hyper)
#     optimizer = opt(best_hyper, lr=lr_outer)

#     best_outer_fold, best_outer_loss = 0, 10
#     # Outer train/test loop
#     for epoch in range(epochs_outer):
#         model_state = train2(dataloader_train_outer, model, loss_fn, optimizer)
#         test_loss, pred_arr = test(dataloader_test_outer, model, loss_fn)

#         # Save prediction array for best epoch of best model
#         if test_loss < best_outer_loss:
#             best_outer_fold, best_outer_loss = outer_layer, test_loss
#             best_predictions[outer_layer] = pred_arr
#             best_loss_list[outer_layer] = best_outer_loss

#         print(f"Epoch: {epoch} - loss: {test_loss}")

#     # Print
#     print("#"*10)
#     print(f"Outer layer: {outer_layer} - best hyper: {best_hyper} - loss: {best_outer_loss}")

# print("\n\n")
# print("-"*10)
# print(f"best_hyper_list:\n{best_hyper_list}")
# print(f"best_loss_list:\n{best_loss_list}")
# print()




