
library("MESS")

x <- rnorm(100, mean = 50, sd=5)



## Define the plotting function
qqwrap <- function(x, y, ...){
  stdy <- (y-mean(y))/sd(y)
  qqnorm(stdy, main="", ...)
  qqline(stdy)}

################################
## Check af normalitetsantagelsen med q-q plots og Wally-plot
## Do the Wally plot
wallyplot(x, FUN=qqwrap, ylim=c(-3,3))



