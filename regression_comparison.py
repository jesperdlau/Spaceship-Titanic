
import random 
import numpy as np
import scipy.stats as st

n = 1000
label  = np.array([random.random() for _ in range(n)])
model1 = np.array([random.random() for _ in range(n)])
model2 = np.array([random.random() for _ in range(n)])
model3 = np.array([random.random() for _ in range(n)])

def ttest_twomodels(y_true, yhatA, yhatB, alpha=0.05, loss_norm_p=1):
    zA = np.abs(y_true - yhatA) ** loss_norm_p
    # Compute confidence interval of z = zA-zB and p-value of Null hypothesis
    zB = np.abs(y_true - yhatB) ** loss_norm_p

    z = zA - zB
    CI = st.t.interval(1 - alpha, len(z) - 1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
    p = 2*st.t.cdf(-np.abs(np.mean(z)) / st.sem(z), df=len(z) - 1)  # p-value
    return np.mean(z), CI, p


def pairwise_stats(label, model1, model2, model3,printStat = True):
    result = [[] for _ in range(3)]
    
    mean12, CI12, p12 = ttest_twomodels(label, model1, model2)
    mean23, CI23, p23 = ttest_twomodels(label, model2, model3)
    mean13, CI13, p13 = ttest_twomodels(label, model1, model3)

    if printStat:
        print("Stats for test between model1 and model2: ")
        print("Mean difference = ",mean12)
        print("Confidence interval for the mean difference: ",CI12)
        print("P-value = ",p12)

        print("\nStats for test between model2 and model3: ")
        print("Mean difference = ",mean23)
        print("Confidence interval for the mean difference: ",CI23)
        print("P-value = ",p23)

        print("\nStats for test between model1 and model3: ")
        print("Mean difference = ",mean13)
        print("Confidence interval for the mean difference: ",CI13)
        print("P-value = ",p13)

    mean_list = [mean13,mean23,mean13]
    CI_list   = [CI12,CI23,CI13]
    p_list    = [p12,p23,p13]

    return mean_list, CI_list, p_list


if __name__ == "__main__":
    # mean, CI, p = ttest_twomodels(label,model1,model2)
    # print("Mean differnce = ", mean)
    # print("Confidence interval for the mean difference: ",CI)
    # print("P-value = ",p)

    pairwise_stats(label,model1,model2,model3)


