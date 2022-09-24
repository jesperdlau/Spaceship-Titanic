# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_naive import df
from sklearn.decomposition import PCA


# Define the spending columns from the importet df
spending = df.iloc[:,9:14]
#print(spending)

# PCA over spending
pca = PCA()
pca.fit(spending)

# Per component- and accumulated variance explained
exp = pca.explained_variance_ratio_
sum_ = 0
acc = []
for val in exp:
    sum_ += val
    acc.append(sum_)

# Plot explained variance
plt.plot(exp, "o")
plt.plot(acc, "-")
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()


