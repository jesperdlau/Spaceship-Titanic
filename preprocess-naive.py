# Import 
# import numpy as np
import pandas as pd
from sklearn import preprocessing

# Helping functions

min_max_scaler = preprocessing.MinMaxScaler()

def mm_scale_column(df, col):
    col_np = df[col].to_numpy()
    col_list = [[val] for val in col_np]
    col_scaled = min_max_scaler.fit_transform(col_list)
    return col_scaled


# Input data csv
filename_train = "Spaceship-Titanic/Data/train.csv"
# filename_test = "./Data/test.csv"

df = pd.read_csv(filename_train)

# PassengerId
# TODO: Ignore for now. Be aware of column indexing when undeleting
del df["PassengerId"]

# HomePlanet
# TODO: Beslut om homeplanets skal være en 3d array eller 3 kollonner

y = pd.get_dummies(df["HomePlanet"], prefix="HomePlanet")
del df["HomePlanet"]

df.insert(0, column="Home_Earth", value=y["HomePlanet_Earth"])
df.insert(1, column="Home_Europa", value=y["HomePlanet_Europa"])
df.insert(2, column="Home_Mars", value=y["HomePlanet_Mars"])

# CryoSleep 
# Fills nan as 0
df["CryoSleep"] = df["CryoSleep"].fillna(0)
df["CryoSleep"] = df["CryoSleep"].astype(int)

# Cabin 
# TODO: Ignored for now. Be aware of column indexing when undeleting
del df["Cabin"]

# Destination
# One-Hot encodes destination
# '55 Cancri e' 'TRAPPIST-1e' 'PSO J318.5-22' 
df.dropna(subset=["Destination"], inplace=True)
h = pd.get_dummies(df["Destination"], prefix="Dest")
del df["Destination"]

df.insert(4, column="Dest_Cancri", value=h["Dest_55 Cancri e"])
df.insert(5, column="Dest_TRAPPIST", value=h["Dest_PSO J318.5-22"])
df.insert(6, column="Dest_PSO", value=h["Dest_TRAPPIST-1e"])

# Age
# Min-Max scaling of age. 
# TODO: problem with outliers and When test data has different range than train data! 
df.dropna(subset=["Age"], inplace=True)
df["Age"] = mm_scale_column(df, "Age")

# VIP
# Fills nan to False/0 because they account for the vast majority
# print(df["VIP"].describe())
df["VIP"] = df["VIP"].fillna(False)
df["VIP"] = df["VIP"].astype(int)


# The following 5 attributes account for spending
# TODO: combined spending attribute?
# Fillna: #  TODO: remove nan instead?
df["RoomService"] = df["RoomService"].fillna(0)
df["FoodCourt"] = df["FoodCourt"].fillna(0)
df["ShoppingMall"] = df["ShoppingMall"].fillna(0)
df["Spa"] = df["Spa"].fillna(0)
df["VRDeck"] = df["VRDeck"].fillna(0)

# Min-Max scale spending attributes:
df["RoomService"] = mm_scale_column(df, "RoomService")
df["FoodCourt"] = mm_scale_column(df, "FoodCourt")
df["ShoppingMall"] = mm_scale_column(df, "ShoppingMall")
df["Spa"] = mm_scale_column(df, "Spa")
df["VRDeck"] = mm_scale_column(df, "VRDeck")

# RoomService
# FoodCourt
# ShoppingMall
# Spa
# VRDeck


# Name
# Ignore name for now.
# TODO: Make family connection?
del df["Name"]

# Transported (Label)
df["Transported"] = df["Transported"].astype(int)


if __name__ == "__main__":
    print(df.describe(include="all"))
    print(df.iloc[:,:5])
    print(df.iloc[:,5:10])
    print(df.iloc[:,10:])
    print(f"\n        Isna: \n{df.isna().any()}")
    print(f"\n        Dtypes: \n{df.dtypes}")



