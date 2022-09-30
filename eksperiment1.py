# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 13:27:33 2022

@author: Rasmus
"""

# import pandas as pd

import numpy as np
# df = pd.read_excel(r"Fysik-projekt-1.xlsx")
# # workbook = xlrd.open_workbook('file_name.xlsx')


a = [2.709,2.806,2.708, 2.904,2.753,2.799,2.729,2.679,2.628,2.64]
m = 0.25022
sin_theta = 0.342897807
cos_theta = 0.9393727128473789
g = 9.82

hz = 800
delta_t = [1.39622,1.34795,1.39936,1.57955,1.4501,1.42474,1.3733,1.26955,1.26941,1.2952]
N = [x * hz for x in delta_t]
total_N = sum(N)


weighted_average = 0
for i in range(len(a)):
    weighted_average += a[i]*N[i]

weighted_average = weighted_average/total_N

def F_gnid(a):
    return -(m*a-m*g*sin_theta)

def F_N(a):
    return m*g*cos_theta

def gnid_koef(F_gnid, F_N):
    return F_gnid/F_N

def map_func(a):
    return F_gnid(a)/F_N(a)

def print_results(i):
    print("F_gnid = ",F_gnid(a[i]))
    print("F_N = ", F_N(a[i]))
    print("Gnidningskoeficient = ",gnid_koef(F_gnid(a[i]),F_N(a[i])))

results = list(map(map_func, a))
# print(min(results),max(results))
# print("Standard deviation = ",np.std(results))
print("Mean = ",np.mean(results))
# # print_results(0)

print("Weighted mean =  ",gnid_koef(F_gnid(weighted_average),F_N(weighted_average)))
# print(gnid_koef(F_gnid(weighted_average),F_N(weighted_average)))



