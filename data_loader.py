
import torch
from torch.utils.data import Dataset
#from torchvision import datasets
#from torchvision.transforms import ToTensor

import os
import pandas as pd


class SpaceshipDataset(Dataset):
    def __init__(self, dataframe):
        x = dataframe.iloc[:,:-1].values
        y = dataframe.iloc[:,-1].values

        self.X=torch.tensor(x, dtype=torch.float32)
        self.Y=torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


if __name__ == "__main__":
    from preprocess import df
    from torch.utils.data import DataLoader
    from sklearn import preprocessing
    from utilities import scale_df

    df = scale_df(df, df, scaler=preprocessing.MinMaxScaler())
    train_data = SpaceshipDataset(df)
    #test_data = SpaceshipDataset(df)
    train_loader=DataLoader(train_data,batch_size=2,shuffle=False)
    #test_loader=DataLoader(train_data,batch_size=2,shuffle=False)

    for i, (data, labels) in enumerate(train_loader):
        print(data.shape, labels.shape)
        print(data,labels)
        break
