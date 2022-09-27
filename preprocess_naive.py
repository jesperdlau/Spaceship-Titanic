# Import 
import numpy as np
import pandas as pd
from sklearn import preprocessing


# Input data csv
filename_train = "Spaceship-Titanic/Data/train.csv"
filename_train_abs = "/home/jesper/Documents/MachineLearning/Spaceship-Titanic/Data/train.csv"
# filename_test = "./Data/test.csv"

# DataFrame
df = pd.read_csv(filename_train)

# PassengerId
# TODO OK Ignore for now. Be aware of column indexing when undeleting
del df["PassengerId"]

# HomePlanet
# TODO OK Beslut om homeplanets skal være en 3d array eller 3 kollonner. Bliver som det er. 

y = pd.get_dummies(df["HomePlanet"], prefix="HomePlanet")
del df["HomePlanet"]

df.insert(0, column="Home_Earth", value=y["HomePlanet_Earth"])
df.insert(1, column="Home_Europa", value=y["HomePlanet_Europa"])
df.insert(2, column="Home_Mars", value=y["HomePlanet_Mars"])

# CryoSleep 
# Removes nan because we have enough data
df.dropna(subset=["CryoSleep"], inplace=True)
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
# Scaling of age. 
# TODO: Is is still better to use min-max scaling? To preserve som range.. 
# One solution. Age above 100 is set to 1. Otherwise set to age*0.01
df.dropna(subset=["Age"], inplace=True)
df["Age"] = df["Age"].apply(lambda x: 0.01*x if x <= 100 else 1.)

# VIP
# Fills nan to False/0 because they account for the vast majority
# print(df["VIP"].describe())
df["VIP"] = df["VIP"].fillna(False)
df["VIP"] = df["VIP"].astype(int)


# The following 5 attributes account for spending
# Fillna: #  TODO OK Fill with average for column. 
df["RoomService"] = df["RoomService"].fillna(df["RoomService"].mean())
df["FoodCourt"] = df["FoodCourt"].fillna(df["FoodCourt"].mean())
df["ShoppingMall"] = df["ShoppingMall"].fillna(df["ShoppingMall"].mean())
df["Spa"] = df["Spa"].fillna(df["Spa"].mean())
df["VRDeck"] = df["VRDeck"].fillna(df["VRDeck"].mean())

# Insert new TotalSpending attribute
TotalSpending = df.loc[:,"RoomService":"VRDeck"].sum(axis=1)
df.insert(15, column="TotalSpending", value=TotalSpending)

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
    print(df["TotalSpending"].where(df["TotalSpending"]==0).count())
    print(df["VIP"].where(df["VIP"]==1).count())

