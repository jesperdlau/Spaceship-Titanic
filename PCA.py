# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocess_naive import df
from sklearn.decomposition import PCA


# Define the spending columns from the importet df
spending = df.loc[:,"RoomService":"VRDeck"]

# PCA over spending
pca = PCA()
pca.fit(spending)

# Per component- and accumulated variance explained
exp = pca.explained_variance_ratio_
cum_sum = np.cumsum(exp)

# Plot explained variance
plt.plot(exp, "o")
plt.plot(cum_sum, "x")
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()


# TODO: Data projected onto PCA



# TODO: Plot of data projected onto PCA

