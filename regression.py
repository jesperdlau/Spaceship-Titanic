# Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from utilities import scale_df, inverse_scale_df, scale_col


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

# Linear Regression
lin_reg = LinearRegression(fit_intercept=True).fit(X_train, y_train)
coef = lin_reg.coef_
intercept = lin_reg.intercept_
#function = f"{intercept:.3f} + {coef[0]:.3f}*RoomService + {coef[1]:.3f}*FoodCourt + {coef[2]:.3f}*ShoppingMall + {coef[3]:.3f}*Spa"
function = f"{intercept:.3f} + {coef[0]:.3f} + {coef[1]:.3f} + {coef[2]:.3f} + {coef[3]:.3f}"

# Linear Regression prediction
pred = lin_reg.predict(X_eval)
pred_true = pd.DataFrame({"Prediction": pred, "True Value": y_eval})
comparison = X_eval.join(pred_true)

# Scoring
MSE = mean_squared_error(y_eval, pred)
r2 = r2_score(y_eval, pred)

# Print output
#print(cost_train[:5])
print(f"MSE: {MSE}")
print(f"r2-score: {r2}")
#print(f"Intercept: {intercept}")
#print(f"Coef: {coef}")
print(f"{function}")
print(comparison[:10])


# Plot output
# Difficult to plot in a simple way when you have 4 parameters
# plt.scatter(X_eval.iloc[:50,0], y_eval[:50], color="black", label="True Value")
# plt.scatter(X_eval.iloc[:50,0], pred[:50], color="blue", label="Prediction")

# for n in range(50):
#     plt.arrow(X_eval.iloc[n,0], y_eval[n], 0, pred[n]-y_eval[n])

# plt.legend()
# plt.show()


