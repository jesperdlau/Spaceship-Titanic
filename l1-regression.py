# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from utilities import scale_df, inverse_scale_df, scale_col

# Hyperparameters

# Prepare data
csv_input_train = "Spaceship-Titanic/Data/train_preprocessed.csv"
csv_input_eval = "Spaceship-Titanic/Data/eval_preprocessed_full.csv"
df_train = pd.read_csv(csv_input_train)
df_eval = pd.read_csv(csv_input_eval)

# ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
# Data split already provided
X_train = df_train.loc[:,"RoomService":"Spa"]
y_train = df_train.loc[:,"VRDeck"]
X_eval = df_eval.loc[:,"RoomService":"Spa"]
y_eval = df_eval.loc[:,"VRDeck"]


# Scaling 
scaler = StandardScaler()
# scaler = MinMaxScaler()
X_eval = scale_df(X_train, X_eval, scaler)
X_train = scale_df(X_train, X_train, scaler)
y_eval = scale_col(y_train, y_eval, scaler)
y_train = scale_col(y_train, y_train, scaler)


def l1_loop(l1alpha):
    l1_reg = Lasso(alpha=l1alpha).fit(X_train, y_train)
    #l1_reg = Ridge(alpha=l1alpha).fit(X_train, y_train)

    coef = l1_reg.coef_
    intercept = l1_reg.intercept_

    # Linear Regression prediction
    pred_eval = l1_reg.predict(X_eval)
    pred_train = l1_reg.predict(X_train)
    # pred_true = pd.DataFrame({"Prediction": pred, "True Value": y_eval})
    # comparison = X_eval.join(pred_true)

    # Scoring
    MSE_eval = mean_squared_error(y_eval, pred_eval)
    r2_eval = r2_score(y_eval, pred_eval)
    MSE_train = mean_squared_error(y_train, pred_train)
    r2_train = r2_score(y_train, pred_train)


    return MSE_train, r2_train, MSE_eval, r2_eval, coef, intercept



MSE_train_list = []
r2_train_list = []
MSE_eval_list = []
r2_eval_list = []
weights = []
intercepts = []

for l in np.linspace(0, 0.3, 200):
    MSE_train, r2_train, MSE_eval, r2_eval, coef, intercept = l1_loop(l)
    MSE_train_list.append(MSE_train)
    r2_train_list.append(r2_train)
    MSE_eval_list.append(MSE_eval)
    r2_eval_list.append(r2_eval)
    weights.append(coef)
    intercepts.append(coef)
    #function = f"{l:.2f}: {intercept:.3f} + {coef[0]:.3f} + {coef[1]:.3f} + {coef[2]:.3f} + {coef[3]:.3f}"
    #print(function)


fig, (ax0, ax1, ax2) = plt.subplots(3,1)

# plot weights
for w in range(4):
    ax0.plot([wei[w] for wei in weights])
    ax0.plot(intercepts[w])
ax0.set_title("Weights")

# Plot score/error
ax1.plot(MSE_train_list, 'b--')
ax1.plot(MSE_eval_list, 'b')
ax1.legend(["MSE_train", "MSE_eval"])
ax1.set_title("L1 regression. Mean Square Error, lower is better")
ax1.set_ylabel("Error")
ax1.set_xlabel("Lambda")

ax2.plot(r2_train_list, 'r--')
ax2.plot(r2_eval_list, 'r')
ax2.legend(["r2_train", "r2_eval"])
ax2.set_title("L1 regression. r2 score - Higher is better. 1.0 is perfect.")
ax2.set_ylabel("Score")
ax2.set_xlabel("Lambda")

plt.show()


