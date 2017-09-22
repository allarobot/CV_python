# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('basil.jpg')
plt.subplot(231)
plt.imshow(img)
plt.title("raw img")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
plt.subplot(232)
plt.imshow(binary)
plt.title("binary")
kernel = np.ones((3,3),np.uint8)
binary = cv2.morphologyEx(binary,cv2.MORPH_OPEN,kernel)
sure_bg = cv2.dilate(binary,kernel,iterations=3)
plt.subplot(233)
plt.imshow(sure_bg)
plt.title("sure back ground")
distanceTransform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(distanceTransform,0.7*distanceTransform.max(),255,cv2.THRESH_BINARY)
sure_fg = np.uint8(sure_fg)
unsure = cv2.subtract(sure_bg,sure_fg)
plt.subplot(234)
plt.imshow(sure_fg)
plt.title("sure front ground")
ret,markers = cv2.connectedComponents(sure_fg)
markers =markers+1
markers[unsure==255]=0
plt.subplot(235)
plt.imshow(markers)
plt.title("markers")
cv2.watershed(img,markers)
img[markers==-1]=[255,0,0]
plt.subplot(236)
plt.imshow(img)
plt.title("segm result")