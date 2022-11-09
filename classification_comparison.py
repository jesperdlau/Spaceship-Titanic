
# import logit_regression

import random 
import numpy as np
# import math
# from scipy import special
from scipy import stats
#from statsmodels.stats.contingency_tables import mcnemar

n = 1000

model1 = [bool(random.getrandbits(1)) for _ in range(n)]
model2 = [bool(random.getrandbits(1)) for _ in range(n)]

n11, n12, n21, n22 = 0,0,0,0

# Calculates the n matrix
for pred in zip(model1,model2):
    if pred[0] == 0:
        if pred[1] == 0:
            n22 += 1
        else:
            n21 += 1
    else:
        if pred[1] == 0:
            n12 += 1
        else:
            n11 += 1

p12 = n12/n
p21 = n21/n

r_hat = n12/(n12 + n21)
r = p12/(p12 + p21)


print("n11,n12,n21,n22 = ", n11,n12,n21,n22)
print("p12 = ",p12)
print("p21 = ",p21)
print("r_hat = ",r_hat)
print("r = ",r)

s = n12 + n21

# P_value McNemars
#p_value = special.binom(s,n12) * r**(n12) * (1-r)**n21
#print("p_value = ",p_value)
p_value = 2 * stats.binom.cdf(k = min(n12,n21),n = n12 + n21, p = 1/2)
print("P_value = ",p_value)

theta_hat = (n12 - n21)/n
print("Theta_hat = ",theta_hat)


# Conf int --- McNemars
E_theta = (n12-n21)/n
Q = (n**2 * (n + 1)*(E_theta + 1)*(1 - E_theta)) / (n*(n12 + n21) - (n12 - n21)**2)
f = (E_theta + 1)/2 * (Q - 1)
g = (1 - E_theta)/2 * (Q - 1)

conf_int = stats.beta.interval(confidence=0.95,a=f,b=g)

print("f = ",f,"g = ",g)
print("Confidence interval for theta: ",conf_int)

