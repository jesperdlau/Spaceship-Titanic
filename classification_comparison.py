
import random 
import numpy as np
# import math
# from scipy import special
import scipy.stats as st
#from statsmodels.stats.contingency_tables import mcnemar

def mcnemar(y_true, yhatA, yhatB, alpha=0.05, print_stats =False):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat)
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in st.beta.interval(1-alpha, a=p, b=q) )

    p = 2*st.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    if print_stats:
        print("Result of McNemars test using alpha=", alpha)
        print("Comparison matrix n")
        print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    if print_stats:
        print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
        print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p

def pairwise_stats_classification(label,model1,model2,model3,print_stats = True):
    theta_hat12, CI12, p12 = mcnemar(label, model1, model2)
    theta_hat23, CI23, p23 = mcnemar(label, model2, model3)
    theta_hat13, CI13, p13 = mcnemar(label, model1, model3)

    if print_stats:
        print("Stats for test between model1 and model2: ")
        print("Theta_hat = ",theta_hat12)
        print("Confidence interval for Theta_hat: ",CI12)
        print("P-value = ",p12)

        print("\nStats for test between model2 and model3: ")
        print("Theta_hat = ",theta_hat23)
        print("Confidence interval for Theta_hat: ",CI23)
        print("P-value = ",p23)

        print("\nStats for test between model1 and model3: ")
        print("Theta_hat = ",theta_hat13)
        print("Confidence interval for Theta_hat: ",CI13)
        print("P-value = ",p13)

    theta_hat_list = [theta_hat13,theta_hat23,theta_hat13]
    CI_list   = [CI12,CI23,CI13]
    p_list    = [p12,p23,p13]

    return theta_hat_list, CI_list, p_list

if __name__ == "__main__":
    n = 1000
    label  = np.array([random.getrandbits(1) for _ in range(n)])
    model1 = np.array([random.getrandbits(1) for _ in range(n)])
    model2 = np.array([random.getrandbits(1) for _ in range(n)])   
    model3 = np.array([1 for _ in range(n)])

    pairwise_stats_classification(label,model1,model2,model3)

# # Own implementation
# n = 1000
# model1 = [bool(random.getrandbits(1)) for _ in range(n)]
# model2 = [bool(random.getrandbits(1)) for _ in range(n)]

# n11, n12, n21, n22 = 0,0,0,0

# # Calculates the n matrix
# for pred in zip(model1,model2):
#     if pred[0] == 0:
#         if pred[1] == 0:
#             n22 += 1
#         else:
#             n21 += 1
#     else:
#         if pred[1] == 0:
#             n12 += 1
#         else:
#             n11 += 1

# p12 = n12/n
# p21 = n21/n

# r_hat = n12/(n12 + n21)
# r = p12/(p12 + p21)

# # print("n11,n12,n21,n22 = ", n11,n12,n21,n22)
# # print("p12 = ",p12)
# # print("p21 = ",p21)
# # print("r_hat = ",r_hat)
# # print("r = ",r)

# s = n12 + n21

# # P_value McNemars
# #p_value = special.binom(s,n12) * r**(n12) * (1-r)**n21
# #print("p_value = ",p_value)
# p_value = 2 * st.binom.cdf(k = min(n12,n21),n = n12 + n21, p = 1/2)
# # print("P_value = ",p_value)

# theta_hat = (n12 - n21)/n
# # print("Theta_hat = ",theta_hat)

# # Conf int --- McNemars
# E_theta = (n12-n21)/n
# Q = (n**2 * (n + 1)*(E_theta + 1)*(1 - E_theta)) / (n*(n12 + n21) - (n12 - n21)**2)
# f = (E_theta + 1)/2 * (Q - 1)
# g = (1 - E_theta)/2 * (Q - 1)

# conf_int = st.beta.interval(confidence=0.95,a=f,b=g)

# # print("f = ",f,"g = ",g)


# print("P_value = ",p_value)

# print("Confidence interval for theta: ",conf_int)
