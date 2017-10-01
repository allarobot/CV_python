#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 23:42:37 2017

@author: appel
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('varese.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
keypoints,descriptor = sift.detectAndCompute(gray,None)
img = cv2.drawKeypoints(img,keypoints,img,[0,0,255],cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("",img)
cv2.imwrite("varese_SIFT.jpg",img)
cv2.waitKey()
cv2.destroyAllWindows()
