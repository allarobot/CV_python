# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:49:50 2017

@author: Administrator
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from scipy.optimize import leastsq



fp = open("Data and Image\\Flatness data_Question3.csv")
lines =csv.reader(fp,delimiter=',')
for row in lines:
    data=row
newdata = []
for num in data:
    newdata.append(float(num))

newdata = np.array(newdata)
z = newdata.reshape(10,250)
x,y = np.mgrid[1:10:10j,1:250:250j]
X = x.flatten()
Y = y.flatten()
Z = z.flatten()
def error(p):
    a,b,c = p
    return a*X+b*Y+c-Z

ax=plt.subplot(111,projection='3d')  
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)

plt.show()
r = leastsq(error,[0,0,1])
A,B,C = r[0]
r = np.sqrt(A**2+B**2+1**2)
d = (A*X+B*Y+C)/r
flattness = max(d)-min(d)
print("flattness",flattness)
