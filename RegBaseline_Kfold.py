from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilities import scale_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


save_path = "Spaceship-Titanic/Data/BaselineReg_pred.npy"
source_path = "Spaceship-Titanic/Data/data_regression.csv"
k_inner = 5
k_outer = 5
test_size = 0.2

# Prepare data
csv_input_train = source_path
# csv_input_eval = source_path
df_train = pd.read_csv(csv_input_train)
# df_eval = pd.read_csv(csv_input_eval)

# Data split
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1]
# Scaling
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# X_train = scale_df(X_train, X_train, scaler)

cv_outer = KFold(n_splits=k_outer,shuffle=False)

outer_results = list()
i = 1
y_hat_list = []
for train_ix, test_ix in cv_outer.split(X_train):
    X_train_inner, X_test_inner = X_train.iloc[train_ix, :], X_train.iloc[test_ix, :]
    y_train_inner, y_test_inner = y_train.iloc[train_ix], y_train.iloc[test_ix]

    cv_inner = KFold(n_splits=k_inner, shuffle=False)

    model = DummyRegressor(strategy = "mean")
    
    #search = GridSearchCV(model,scoring="neg_mean_squared_error",n_jobs=1,cv=cv_inner,refit=True)

    #result = search.fit(X_train_inner,y_train_inner)
    result = model.fit(X_train_inner,y_train_inner)
    # best_model = result.best_estimator_

    y_hat = result.predict(X_test_inner)
    
    y_hat_list.append(y_hat)

    error = mean_squared_error(y_test_inner,y_hat)

    outer_results.append(error)
    # print('>MSE=%.3f, est=%.3f, opt_alpha=%s' % (error, result.best_score_, result.best_params_))
    print(f"Fold {i}: ",error)
    i+=1

# scores = cross_val_score(search,X_train,y_train,scoring="neg_mean_squared_error",cv=cv_outer,n_jobs=1,error_score="raise")
# print(scores)

# arr = np.array(y_hat_list)
# np.save(save_path,arr)

print('Mean_MSE: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))