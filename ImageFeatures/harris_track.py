#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:05:04 2017

@author: appel
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#img = cv2.imread('track_small.jpg')
img = cv2.imread('track.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray_f = np.float32(gray)
dst = cv2.cornerHarris(gray_f,10,23,0.04)
roi = dst>0.01*dst.max()
img[roi]=[0,0,255] 
plt.imshow(img)
#cv2.imshow('corners',img)
#cv2.waitKey()
#cv2.destroyAllWindows()