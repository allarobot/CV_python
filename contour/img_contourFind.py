# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np

img = np.zeros((200,200),dtype=np.uint8)
img[50:60,50:60] = 200
img[60:70,60:70] = 255
img[70:80,70:80] = 175
cv2.imshow("test",img)
ret,thresh = cv2.threshold(img,127,255,0)
cv2.imshow("binary",thresh)
image,contours,hier=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("contour_image",image)
color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
cv2.imshow("color",color)
##contour of object
newcolor=cv2.drawContours(color,contours,-1,(0,0,255))
cv2.imshow("newcolor",newcolor)
i=0
for c in contours:
    print("ct:",c)
    i +=1
    #rectangle contour
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(color,(x,y),(x+w,y+h),(0,255,0),2)
    
    #min rectangle contour
    rect = cv2.minAreaRect(c)
    print("rect",rect)
    box = cv2.boxPoints(rect)
    print("box",box)
    box = np.int0(box)
    cv2.drawContours(color,[box],-1,(0,0,255),3)
cv2.imshow("contours",color)    

cv2.waitKey()
cv2.destroyAllWindows()
