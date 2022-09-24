#Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_naive import df

# Define the spending columns from the importet df
spending = df.iloc[:,9:14]

# df["RoomService"]
# df["FoodCourt"]
# df["ShoppingMall"]
# df["Spa"]
# df["VRDeck"]

# Covariance and correlation
cov = spending.cov()
corr = spending.corr()

# Plot Pandas
# Pandas Plot virker ikke uden HTML, CSS
#corr.style.background_gradient(cmap='coolwarm')
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps

# Plot plt
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns)
# plt.yticks(range(len(corr.columns)), corr.columns)
# plt.show()

# Plot Seaborn/plt
# sns.heatmap(cov, 
#             xticklabels=cov.columns.values,
#             yticklabels=cov.columns.values)
# plt.title("Covariance Matrix")
# plt.show()

sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title("Covariance Matrix")
plt.show()

