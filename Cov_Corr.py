#Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_naive import df

# Define the spending columns from the importet df
spending = df.loc[:,"RoomService":"VRDeck"]

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

# Plot 
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.matshow(corr)
# plt.xticks(range(len(corr.columns)), corr.columns)
# plt.yticks(range(len(corr.columns)), corr.columns)
# plt.show()

# Plot Seaborn heatmap for covariance and correlation
# sns.heatmap(cov, 
#             xticklabels=cov.columns.values,
#             yticklabels=cov.columns.values)
# plt.title("Covariance Matrix")
# plt.show()
#
# sns.heatmap(corr, 
#             xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.title("Correlation Matrix")
# plt.show()


#Plot VIP against TotalSpending with Seaborn boxplot/violinplot
figure, axs = plt.subplots(1, 2, figsize=(6, 12))
axs[0].set_yscale("log")
axs[1].set_yscale("log")
figure.suptitle("VIP against TotalSpending - Logarithmic y-axis")
axs[0].set_title("Boxplot")
axs[1].set_title("Violinplot")
sns.boxplot(ax=axs[0], data=df, x="VIP", y="TotalSpending")
sns.violinplot(ax=axs[1], data=df, x="VIP", y="TotalSpending")
plt.show()


# Plot VIP agains 4 spending
# fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# axs[0,0].scatter(df["VIP"], df["FoodCourt"])
# axs[1,0].scatter(df["VIP"], df["ShoppingMall"])
# axs[0,1].scatter(df["VIP"], df["Spa"])
# axs[1,1].scatter(df["VIP"], df["VRDeck"])
# plt.show()

# Plot Seaborn Pairplot
# Is slow! ... and doesn't tell much. 
# TODO: Try different scalings for more informative pairplot. 
# sns.pairplot(data=df, 
#             x_vars=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], 
#             y_vars=["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"], 
#             hue="VIP",
#             kind="kde",
#             #kind="heat",
#             )
# plt.title("Pair Plot")
# plt.show()


# df["RoomService"]
# df["FoodCourt"]
# df["ShoppingMall"]
# df["Spa"]
# df["VRDeck"]

