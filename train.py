# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# Import from other files
from data_loader import SpaceshipDataset
from preprocess import df
from utilities import scale_df

# Hyperparameters 
test_size= 0.25
random_state= 42
batch_size = 5
lr = 1e-4
epochs = 10

# Prepare data
# Scale data here instead of preprocess as data may be used with other scalars in other parts of the project. 
mm_scale = preprocessing.MinMaxScaler()
df_scaled = scale_df(df, df, mm_scale)
df_train, df_test = train_test_split(df_scaled, test_size=test_size, random_state=random_state)
dataset_train, dataset_test = SpaceshipDataset(df_train), SpaceshipDataset(df_test)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Model
from model import NeuralNetwork
model = NeuralNetwork()

# Loss function and Optimizer
# Binary Cross Entropy is standard for binary classification
loss_fn = nn.BCEWithLogitsLoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print info for batch
        if batch_nr % 100 == 0:
            loss, current = loss.item(), batch_nr * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.unsqueeze(1).to(device)

            pred = model(X)
            pred_bin = torch.round(torch.sigmoid(pred))

            test_loss += loss_fn(pred, y).item()
            correct += (pred_bin == y).type(torch.float).sum().item()
    avg_loss = test_loss/num_batches
    accuracy = correct/size
    #print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return accuracy, avg_loss

if __name__ == "__main__":
    acc_list = []
    loss_list = []
    for t in range(epochs):
        
        train(dataloader_train, model, loss_fn, optimizer)
        accuracy, avg_loss = test(dataloader_test, model, loss_fn)

        acc_list.append(accuracy)
        loss_list.append(avg_loss)

        print(f"Epoch {t+1}\n-------------------------------")
        print(f"Test Error: \nAccuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")

    print("Done!")
    plt.plot(acc_list)
    plt.ylim((0,1))
    plt.show()