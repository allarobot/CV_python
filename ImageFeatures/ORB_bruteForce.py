#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 01:11:12 2017

@author: appel
"""

import cv2
img1 = cv2.imread('manowar_logo.png')
img2 = cv2.imread('manowar_single.jpg')
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1,des1 = orb.detectAndCompute(gray1,None)
kp2,des2 = orb.detectAndCompute(gray2,None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches,key=lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:40],img2,flags=2)
cv2.imshow("",img3)
cv2.imwrite("manowar_ORB_BF.png",img3)
cv2.waitKey()
cv2.destroyAllWindows()