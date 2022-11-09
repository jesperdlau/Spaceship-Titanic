
import random 
import numpy as np
import scipy.stats as st

n = 1000
label = np.array([random.random() for i in range(n)])
model1 = np.array([random.random() for i in range(n)])
model2 = np.array([random.random() for i in range(n)])


def ttest_twomodels(y_true, yhatA, yhatB, alpha=0.05, loss_norm_p=1):
    zA = np.abs(y_true - yhatA) ** loss_norm_p
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zB = np.abs(y_true - yhatB) ** loss_norm_p

    z = zA - zB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return np.mean(z), CI, p

mean, CI, p = ttest_twomodels(label,model1,model2, alpha = 0.05)
print("mean = ", mean)
print("Confidence interval for the mean: ",CI)
print("P-value = ",p)

