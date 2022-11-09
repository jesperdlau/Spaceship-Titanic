
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn import preprocessing


# Import from other files
from data_loader import SpaceshipDataset, EvalLoader
#from preprocess import df
from utilities import scale_df
from model import ClassificationModel


# Hyperparameters 
# csv_path = ""
# eval_df_path = ""
scaler = preprocessing.MinMaxScaler()
state_dict_path = "model_3.pth"
batch_size = 1
csv_input_eval = "Spaceship-Titanic/Data/eval_preprocessed_full.csv"
csv_input_train = "Spaceship-Titanic/Data/train_preprocessed.csv"
pred_out_path = "space_pred.csv"

# Prepare data
# Removes Id column as model is not trained for it. But saves it for later merge.
df_eval = pd.read_csv(csv_input_eval)
df_train = pd.read_csv(csv_input_train)
df_id = df_eval["PassengerId"]
del df_eval["PassengerId"]
del df_train["PassengerId"]

# Prepare data and dataloader
# Scale data here instead of preprocess as data may be used with other scalars in other parts of the project. 
df_scaled = scale_df(df_train.iloc[:,:-1], df_eval, scaler)
dataset_eval = EvalLoader(df_scaled)
dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=False)


# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Prepare model
model = ClassificationModel()
state_dict = torch.load(state_dict_path)
model.load_state_dict(state_dict)

# Evaluation loop 
def evaluate(dataloader, model):
    prediction_list = []
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)

            pred = model(X)
            pred_bin = torch.round(torch.sigmoid(pred))
            #pred_bin.detach().numpy()
            #pred_bin = pred_bin.cpu().detach().numpy()
            #pred_bin = np.squeeze(pred_bin)
            pred_bin = pred_bin.item()
            prediction_list.append(pred_bin)

    return prediction_list


# Run prediction
prediction_list = evaluate(dataloader_eval, model)
pred_bool = [bool(pred) for pred in prediction_list]
pred_df = pd.DataFrame({"PassengerId": df_id.values, "Transported": pred_bool})

# Save to csv
pred_df.to_csv(pred_out_path, index=False)
print("Done")


