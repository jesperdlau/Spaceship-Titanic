from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from utilities import scale_df
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


source_path = "Spaceship-Titanic/Data/data_classification.csv"
# Number of random trials
# NUM_TRIALS = 5
k_inner = 5
k_outer = 5
test_size = 0.2
lambda_values = np.array([0.00001,0.0001,0.001,0.01,0.02,0.0015])
c_values = np.reciprocal(lambda_values)


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

print(X_train)

# Scaling
scaler = MinMaxScaler()
X_test = scale_df(X_train, X_test, scaler)
X_eval = scale_df(X_train, X_eval, scaler)
X_train = scale_df(X_train, X_train, scaler)

cv_outer = KFold(n_splits=k_outer,shuffle=False)

outer_results = list()
for train_ix, test_ix in cv_outer.split(X_train):
    X_train_inner, X_test_inner = X_train.iloc[train_ix, :], X_train.iloc[test_ix, :]
    y_train_inner, y_test_inner = y_train.iloc[train_ix], y_train.iloc[test_ix]

    cv_inner = KFold(n_splits=k_inner, shuffle=False)

    model = LogisticRegression(penalty="l1",
                solver="liblinear",
                max_iter=10000)
    

    space = dict()
    space['C'] = c_values

    search = GridSearchCV(model,space,scoring="accuracy",n_jobs=1,cv=cv_inner,refit=True)

    result = search.fit(X_train_inner,y_train_inner)

    best_model = result.best_estimator_

    y_hat = result.predict(X_test_inner)
    
    acc = accuracy_score(y_test_inner,y_hat)

    outer_results.append(acc)

    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

scores = cross_val_score(search,X_train,y_train,scoring="accuracy",cv=cv_outer,n_jobs=1,error_score="raise")

print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))