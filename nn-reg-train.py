# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, L1Loss, MSELoss
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Import from other files
from data_loader import SpaceshipDataset, SpaceData
from utilities import scale_df, scale_col
from model import RegressionModel

# Hyperparameters 
test_size= 0.25
random_state= 42
batch_size = 10
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
X_train = df_train.loc[:,:"Spa"]
y_train = df_train.loc[:,"VRDeck"]
X_eval = df_eval.loc[:,:"Spa"]
y_eval = df_eval.loc[:,"VRDeck"]

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
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    epoch_avg_loss = train_loss/len(dataloader_train)
    return epoch_avg_loss


# Test function
def test(dataloader, model, loss_fn):
    model.eval()
    test_loss = 0
    pred_arr = np.zeros(len(dataloader))
    #pred_list = []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.unsqueeze(1).to(device)

            pred = model(X)

            # Save loss
            test_loss += loss_fn(pred, y).item()

            # Save prediction
            #pred_list.append([p.item() for p in pred])
            #prediction_arr[i] = np.array(pred.item())
            pred = np.array([p.item() for p in pred])
            #batch_index = [i+n for n in range(batch_size)]
            #pred_arr = np.insert(pred_arr, batch_index, pred, )
            pred_arr[i] = pred.item()
            #pred_arr.insert(i*batch_size, )

    epoch_avg_loss = test_loss/len(dataloader)
    return epoch_avg_loss, pred_arr

if __name__ == "__main__":
    
    train_loss_list = []
    test_loss_list = []
    pred_list = []

    for t in range(epochs):
            print(f"-------------------------------\nEpoch {t+1}:")
            train_loss = train(dataloader_train, model, loss_fn, optimizer)
            test_loss, pred_arr = test(dataloader_eval, model, loss_fn)

            train_loss_list.append(train_loss)
            test_loss_list.append(test_loss)
            pred_list.append(pred_arr)

    print("Done!")
    #best_model_wts = copy.deepcopy(model.state_dict())

    #Plot 
    plt.plot(train_loss_list, "b--")
    plt.plot(test_loss_list, "r--")
    # plt.ylim((0,1))
    plt.legend(["train_loss", "test_loss"])
    plt.show()