#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 16:25:07 2017

@author: appel
"""
import cv2
import numpy as np

color = cv2.imread('th.jpeg')
gray = cv2.cvtColor(color,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,80,120)
cv2.imshow("color",color)
cv2.imshow("gray",gray)
cv2.imshow("edge",edges)
lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength=10,maxLineGap=5)
circle = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,120,param1=100,param2=30,minRadius=0,maxRadius=0)
#print(lines)
for line in lines:
    x1,y1,x2,y2 = line[0]
    print("line:",x1,y1,x2,y2)
    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)
cv2.circle(color,(circle[0][0][0],circle[0][0][1]),circle[0][0][2],(0,255,0),3)
cv2.imshow("color_with_lines",color)
cv2.waitKey()
cv2.destroyAllWindows()