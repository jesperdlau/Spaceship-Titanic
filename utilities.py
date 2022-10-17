# Import
import numpy as np
import pandas as pd
from sklearn import preprocessing
from preprocess import df


# The most cursed way to scale a column, because for reasons, simply to scaler.transform(df) doesn't work in our case. 
def scale_col(col_train, col_target, scaler):
    column_name = col_train.name
    col_train = col_train.to_numpy()
    col_train = [[val] for val in col_train]
    scaler.fit(col_train)
    col_target = col_target.to_numpy()
    col_target = [[val] for val in col_target]
    col_scaled = scaler.transform(col_target)
    col_scaled = col_scaled.reshape(-1)
    col_scaled = pd.Series(data=col_scaled, name=column_name)
    return col_scaled

# The most cursed way to scale a df, because for reasons, simply to scaler.transform(df) doesn't work in our case. 
def scale_df(df_train, df_target, scaler):
    df_scaled = []
    for col in df_train:
        col_train = df_train[col].to_numpy()
        col_train = [[val] for val in col_train]
        scaler.fit(col_train)
        col_target = df_target[col].to_numpy()
        col_target = [[val] for val in col_target]
        col_scaled = scaler.transform(col_target)
        df_scaled.append(col_scaled)
    df_scaled = np.array(df_scaled)
    df_scaled = df_scaled.T
    df_scaled = df_scaled.reshape(-1, df_scaled.shape[-1])
    df_scaled = pd.DataFrame(data=df_scaled, columns=df_train.columns)
    return df_scaled

if __name__ == "__main__":
    std_scale = preprocessing.StandardScaler()
    mm_scale = preprocessing.MinMaxScaler()
    normalize = preprocessing.Normalizer()  
    
    spending = df.loc[:,"RoomService":"TotalSpending"]
    spending_scaled = scale_df(spending, spending, mm_scale)
    print(spending_scaled)

    scaled_spa = scale_col(df["Spa"], df["Spa"], mm_scale)
    print(scaled_spa)
