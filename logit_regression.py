# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from utilities import scale_df

# Hyperparameters
pred_out_path = "logits_prediction.csv"
test_size = 0.2

# Prepare data
csv_input_train = "Spaceship-Titanic/Data/train_preprocessed.csv"
csv_input_eval = "Spaceship-Titanic/Data/eval_preprocessed_full.csv"
df_train = pd.read_csv(csv_input_train)
df_eval = pd.read_csv(csv_input_eval)

# Data split
df_train, df_test = train_test_split(df_train, test_size=test_size)
X_train = df_train.iloc[:,1:-1]
y_train = df_train.iloc[:,-1]
X_test = df_test.iloc[:,1:-1]
y_test = df_test.iloc[:,-1]
X_eval = df_eval.iloc[:,1:]

# Scaling
scaler = MinMaxScaler()
X_test = scale_df(X_train, X_test, scaler)
X_eval = scale_df(X_train, X_eval, scaler)
X_train = scale_df(X_train, X_train, scaler)

# Logits Regression
log_reg = LogisticRegression().fit(X_train, y_train)


# Prediction
pred_test = log_reg.predict(X_test)
pred_prob_test = log_reg.predict_proba(X_test)

pred_eval = log_reg.predict(X_eval)
pred_prob_eval = log_reg.predict_proba(X_eval)

pred_true = pd.DataFrame({"Prediction": pred_test, "True Value": y_test})


# Scoring
acc = log_reg.score(X_test, y_test)
acc = log_reg.score(X_train, y_train)

# Print output
print(f"Accuracy: {acc}")
print(f"{pred_true.iloc[:10,:]}")


pred_bool = [bool(pred) for pred in pred_eval]
pred_df = pd.DataFrame({"PassengerId": df_eval.iloc[:,0].values, "Transported": pred_bool})

# Save to csv
pred_df.to_csv(pred_out_path, index=False)
print("Done")

