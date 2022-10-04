library("MESS")
library(tidyverse)
library(ggpubr)
library(cowplot)
library(corrplot)

# load data
currentPath = getwd()
df <- read_csv("Dataframe.csv")

# til at se dataset
view(df)
summary(df)

# find number if trues in binary data
n_home_earth  <- sum(df$Home_Earth)
n_home_europa <- sum(df$Home_Europa)
n_home_mars   <- sum(df$Home_Mars)
n_cryosleep   <- sum(df$CryoSleep)
n_transported <- sum(df$Transported)


# Find procentage of zeroes in continuous data
N <- length(df$TotalSpending)
percent_0_roomService   <- length(df$RoomService[df$RoomService==0])/N
percent_0_foodCourt     <- length(df$FoodCourt[df$FoodCourt==0])/N
percent_0_shoppingMall  <- length(df$ShoppingMall[df$ShoppingMall==0])/N
percent_0_spa           <- length(df$Spa[df$Spa==0])/N
percent_0_VRDeck        <- length(df$VRDeck[df$VRDeck==0])/N
percent_0_totalSpending <- length(df$TotalSpending[df$TotalSpending==0])/N

# and for age
n_0yearOlds <- length(df$Age[df$Age==0])
age_noZero <- df$Age[df$Age!=0]

# qq plot of foodcourt spending
ageQQ <- ggqqplot(df$Age*100)
ageDensity <- ggdensity(df$Age*100,"Age")
foodCourtQQ <- ggqqplot(df$FoodCourt)
foodCourtDensity <- ggdensity(df$FoodCourt,xlab="Spending")

plot_grid(foodCourtDensity, ageDensity,foodCourtQQ, ageQQ,ncol=2, nrow=2,label_size = 10
,labels=c("Density plot of foodcourt spending","Density plot of ages","QQ-plot of foodcourt spending","QQ-plot of Age"))

# correlation diagram
corrplot(cor(df[c("VIP","Home_Earth","Home_Europa","Home_Mars",
        "CryoSleep","Age","RoomService",
      "FoodCourt","ShoppingMall","Spa","VRDeck"
      ,"TotalSpending","Transported")]),type = "upper",,method="pie")
