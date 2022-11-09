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



# Import from other files
from data_loader import SpaceshipDataset, SpaceData
from utilities import scale_df, scale_col, inverse_scale_df
from model import RegressionModel

# Hyperparameters 
test_size= 0.25
random_state= 42
batch_size = 2
lr = 1e-4
epochs = 5
scaler = StandardScaler()

# Prepare data
csv_input_train = "Spaceship-Titanic/Data/train_preprocessed.csv"
csv_input_eval = "Spaceship-Titanic/Data/eval_preprocessed_full.csv"
df_train = pd.read_csv(csv_input_train)
df_eval = pd.read_csv(csv_input_eval)

# ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
# Data split already provided
X_train = df_train.loc[:,"Home_Earth":"Spa"]
y_train = df_train.loc[:,"VRDeck"]
X_eval = df_eval.loc[:,"Home_Earth":"Spa"]
y_eval = df_eval.loc[:,"VRDeck"]

# Save raw y for later inverse scaling
y_eval_raw = y_eval

# Scaling 
X_eval = scale_df(X_train, X_eval, scaler)
X_train = scale_df(X_train, X_train, scaler)
y_eval = scale_col(y_train, y_eval, scaler)
y_train = scale_col(y_train, y_train, scaler)

# Data Loader
data_train, data_eval = SpaceData(X_train, y_train), SpaceData(X_eval, y_eval)
dataloader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dataloader_eval = DataLoader(data_eval, batch_size=1, shuffle=False)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Model
model = RegressionModel()

# Loss function and Optimizer
loss_fn = MSELoss()
optimizer = Adam(model.parameters(), lr=lr)


# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss = 0
    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss
        train_loss += loss.item()

        # Print info for batch
        if batch_nr % 50 == 0:
            loss, current = loss.item(), batch_nr * len(X)
            print(f"Train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    epoch_avg_loss = train_loss/len(dataloader_train)
    return epoch_avg_loss, model.state_dict()


# Test function
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    pred_arr = np.zeros(len(dataloader))
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(1).to(device)
            pred = model(X)

            # Save loss
            test_loss += loss_fn(pred, y).item()

            # Save prediction
            pred = np.array([p.item() for p in pred])
            pred_arr[i] = pred.item()

    epoch_avg_loss = test_loss/len(dataloader)
    return epoch_avg_loss, pred_arr

if __name__ == "__main__":
    
    train_loss_list = []
    test_loss_list = []
    pred_list = []
    acc_list = []

    best_loss = 10
    best_model = None
    best_epoch = 0
    best_acc = 0

    for t in range(epochs):
            print(f"-------------------------------\nEpoch {t+1}:")
            train_loss, model_state = train(dataloader_train, model, loss_fn, optimizer)
            test_loss, pred_arr = test(dataloader_eval, model, loss_fn)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            pred_list.append(pred_arr)

            # Accuracy
            #acc = mean_squared_error(y_eval, pred_arr)
            acc = r2_score(y_eval, pred_arr)
            acc_list.append(acc)

            if test_loss < best_loss:
                best_loss = test_loss
                best_model = deepcopy(model_state)
                best_epoch = t
                best_acc = acc


    print("Done!")
    print(f"Best epoch: {best_epoch} with loss: {best_loss} acc: {best_acc}")

    # Inverse scaling and comparison
    best_pred = np.array(pred_list[best_epoch]).reshape(-1,1)
    y_eval_raw
    scaler.fit(y_eval_raw.values.reshape(-1,1))
    y_pred_raw = scaler.inverse_transform(best_pred).squeeze()
    comparison = pd.DataFrame({"True": y_eval_raw.values, "Pred": y_pred_raw})
    print()
    print(comparison.iloc[:10,:])

    raw_score = r2_score(y_eval_raw, y_pred_raw)
    print(f"Raw score: {raw_score}")
    
    # #Plot 
    # plt.plot(train_loss_list, "b--", label="Train Loss")
    # plt.plot(test_loss_list, "r--", label="Test Loss")
    # plt.plot(acc_list, "b", label="Accuracy")
    # # plt.ylim((0,1))
    # plt.legend()
    # plt.show()