# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:49:50 2017

@author: Administrator
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

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

ax=plt.subplot(111,projection='3d')  
ax.plot_surface(x,y,z,rstride=2,cstride=1,cmap=plt.cm.coolwarm,alpha=0.8)

plt.show()
