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
model_ref = RegressionModelHyper1()
optimizer = Adam(model_ref.parameters(), lr=lr)


# Prepare data
data = pd.read_csv(source_path)
# data = df
dataset = SpaceshipDataset(data)

# Outer layer - Important to not shuffle so we can reproduce the splits. The data is already shuffeled. 
kfold = KFold(n_splits=k_folds, shuffle=False)
for outer_layer, (train_id, test_id) in enumerate(kfold.split(data)):


    # Inner layer
    for inner_layer, (train_id_inner, test_id_inner) in enumerate(kfold.split(train_id)):
        dataloader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_id_inner)
        dataloader_test = DataLoader(dataset, batch_size=batch_size, sampler=test_id_inner)


        # Hyper parameter layer
        for hyper in range(hyper_range):
            
            model = RegressionModelHyper1(hyper=hyper)
            best_hyper_state, best_hyper_loss = 0, 10
            
            # Epochs of training
            for epoch in range(epochs):
                
                model_state = train2(dataloader_train, model, loss_fn, optimizer)
                test_loss = test2(dataloader_test, model, loss_fn)

                print(f"[{outer_layer+1}/{k_folds}:{inner_layer+1}/{k_folds}:{hyper+1}/{hyper_range}:{epoch+1}/{epochs}]")

                if test_loss < best_hyper_loss:
                    best_hyper_model, best_hyper_loss = model_state, test_loss



    print()

    




