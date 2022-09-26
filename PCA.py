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

# Plot explained variancefig, ax = plt.subplots()

# target_classes = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# colors = ("blue", "red", "green", "yellow", "black")
# markers = ("^", "s", "o", "x", ".")

# for target_class, color, marker in zip(target_classes, colors, markers):
#     ax.scatter(
#         #x=spending_transformed[target_class],
#         y=spending_transformed[target_class],
#         color=color,
#         label=f"class {target_class}",
#         alpha=0.5,
#         marker=marker,
#     )
# ax.set_title("Spending after PCA")
# plt.show()


plt.plot(exp, "o")
plt.plot(cum_sum, "x")
plt.ylim([0, 1])
plt.ylabel('Explained Variance')
plt.xlabel('Components')
plt.show()


# TODO: Data projected onto PCA
#spending_transformed = pca.transform(spending["RoomService"])

#pca.components_[0]

# plt.scatter(spending_transformed)
# plt.show()

# fig, ax = plt.subplots()

# target_classes = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
# colors = ("blue", "red", "green", "yellow", "black")
# markers = ("^", "s", "o", "x", ".")

# for target_class, color, marker in zip(target_classes, colors, markers):
#     ax.scatter(
#         #x=spending_transformed[target_class],
#         y=spending_transformed[target_class],
#         color=color,
#         label=f"class {target_class}",
#         alpha=0.5,
#         marker=marker,
#     )

# ax.set_title("Spending after PCA")
# plt.show()


# TODO: Plot of data projected onto PCA

