# Credit
# Partial credit to 
# Vanderplas, J.T. (2017) Python Data Science Handbook: Essential Tools for working with data. Sebastopol etc.: O'Reilly. 
# (Vanderplas, Python Data Science Handbook: Essential Tools for working with data 2017)
# https://jakevdp.github.io/PythonDataScienceHandbook/06.00-figure-code.html#Principal-Components-Rotation

# Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from preprocess_naive import df
from utilities import scale_df


# Define the spending columns from the importet df
# Apply standard scaler (Mean 0, sd 1)
spending = df.loc[:,"RoomService":"VRDeck"]
spending = scale_df(spending, spending, scaler=StandardScaler())

# PCA over spending
pca = PCA()
pca.fit(spending)

# Per component- and accumulated variance explained
exp = pca.explained_variance_ratio_
cum_sum = np.cumsum(exp)

# Plot explained variance
# plt.plot(exp, "-")
# plt.plot(cum_sum, "-")
# plt.ylim([0, 1])
# plt.ylabel('Explained Variance')
# plt.xlabel('Components')
# plt.title("Explained Variance")
# plt.legend(["Explained Variance per Component", "Accumulated Explained Variance"])
# plt.show()



# TODO: Plot of PCA component directions
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1[:2], v0[:2], arrowprops=arrowprops)

X = spending.to_numpy()
pca = PCA(n_components=2, whiten=True)
pca.fit(X)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

# plot data
ax[0].scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v, ax=ax[0])
#ax[0].axis('equal');
ax[0].set(xlabel='RoomService', ylabel='FoodCourt', title='Data Input',
        xlim=(-1, 10), ylim=(-1, 10))

# plot principal components
X_pca = pca.transform(X)
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.2)
draw_vector([0, 0], [0, 3], ax=ax[1])
draw_vector([0, 0], [3, 0], ax=ax[1])
#ax[1].axis('equal')
ax[1].set(xlabel='Component 1', ylabel='Component 2',
          title='Data projected onto 2 first Principal Components',
          xlim=(-1, 10), ylim=(-1, 10)
          )

plt.show()
