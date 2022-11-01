# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing


# Import from other files
from data_loader import SpaceshipDataset
from preprocess import df
from utilities import scale_df
from model import NeuralNetwork


# Hyperparameters 
test_size= 0.25
random_state= 42
batch_size = 5
lr = 1e-4
epochs = 20
k_folds = 5
scaler = preprocessing.MinMaxScaler()

# Prepare data
# Scale data here instead of preprocess as data may be used with other scalars in other parts of the project. 
df_scaled = scale_df(df, df, scaler)
data = SpaceshipDataset(df_scaled)
# df_train, df_test = train_test_split(df_scaled, test_size=test_size, random_state=random_state)
# dataset_train, dataset_test = SpaceshipDataset(df_train), SpaceshipDataset(df_test)


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Loss function
loss_fn = nn.BCEWithLogitsLoss()

# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, train_correct = 0, 0
    for batch_nr, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.unsqueeze(1).to(device)

        # Compute prediction error
        pred = model(X)
        pred_bin = torch.round(torch.sigmoid(pred))
        loss = loss_fn(pred, y)

        # Backpropagationdataloader_test
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 
        train_loss += loss.item()
        train_correct += (pred_bin == y).type(torch.float).sum().item()

        # Print info for batch
        if batch_nr % 100 == 0:
            loss, current = loss.item(), batch_nr * len(X)
            print(f"train_loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return train_loss, train_correct


# Test function
def test(dataloader, model, loss_fn):
    #size = len(dataloader.dataset)
    #num_batches = len(dataloader)
    model.eval()
    test_loss, test_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.unsqueeze(1).to(device)

            pred = model(X)
            pred_bin = torch.round(torch.sigmoid(pred))

            test_loss += loss_fn(pred, y).item()
            test_correct += (pred_bin == y).type(torch.float).sum().item()
    #avg_loss = test_loss/num_batches
    #accuracy = correct/size
    #print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    return test_loss, test_correct



### Main: 

fold_train_acc_list = []
fold_test_acc_list = []
fold_train_loss_list = []
fold_test_loss_list = []

# Modified dataloader with k-fold

kfold = KFold(n_splits=k_folds, shuffle=True)
# dataset_train, dataset_test = SpaceshipDataset(df_train), SpaceshipDataset(df_test)


for fold, (train_id, test_id) in enumerate(kfold.split(df_scaled)):
    print(f"*********************************\nFold: {fold+1}\n")
    #train_sampler = SubsetRandomSampler(train_id)
    #test_sampler = SubsetRandomSampler(test_id)

    # Create dataloaders with fold-samples
    # dataloader_train = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
    # dataloader_test = DataLoader(data, batch_size=batch_size, sampler=test_sampler)
    dataloader_train = DataLoader(data, batch_size=batch_size, sampler=train_id)
    dataloader_test = DataLoader(data, batch_size=batch_size, sampler=test_id)

    # Initialize Model and Optimizer for each fold
    model = NeuralNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Set fold accuracy 0 to find best model in fold during loop
    fold_acc = 0
    fold_best_model = 0

    # Save loss and acc for fold
    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    for epoch in range(epochs):
        print(f"-------------------------------\nEpoch {epoch+1}:")
        train_loss, train_correct = train(dataloader_train, model, loss_fn, optimizer)
        test_loss, test_correct = test(dataloader_test, model, loss_fn)

        train_avg_loss = train_loss/len(train_id/batch_size)
        train_accuracy = train_correct/len(train_id)
        test_avg_loss = test_loss/len(test_id/batch_size)
        test_accuracy = test_correct/len(test_id)

        # Store accuracy and loss
        train_acc_list.append(train_accuracy)
        train_loss_list.append(train_avg_loss)
        test_acc_list.append(test_accuracy)
        test_loss_list.append(test_avg_loss)

        # Find best model
        if test_accuracy > fold_acc:
            fold_acc = test_accuracy
            fold_best_model_state = model.state_dict()
        
        #print(f"Test Error: \nAccuracy: {(100*accuracy):>0.1f}%, Avg loss: {avg_loss:>8f} \n")
    
    # Save best model for fold
    torch.save(fold_best_model_state, f"model_{fold}.pth")
    # model.load_state_dict(torch.load(PATH))

    # Save loss and accuracy
    fold_train_acc_list.append(train_acc_list)
    fold_test_acc_list.append(test_acc_list)
    fold_train_loss_list.append(train_loss_list)
    fold_test_loss_list.append(test_loss_list)

    print(f"Fold {fold} - Done! Best accuracy: {fold_acc}")

print(f"###################################\nAll done!")


## Plot test loss and acc for each fold
legend_list = []
for n in range(k_folds):
    plt.plot(fold_test_acc_list[n])
    plt.plot(fold_test_loss_list[n], "--")
    #plt.plot(fold_train_acc_list[n], "-.")
    #plt.plot(fold_train_loss_list[n], ":")
    legend_list.append(f"test_acc {n}")
    legend_list.append(f"test_loss {n}")

# plt.ylim((0,1))
plt.legend(legend_list)
plt.show()