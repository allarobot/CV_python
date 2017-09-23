 # -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 02:23:57 2017

@author: Administrator
"""

import cv2
import matplotlib.pyplot as plt
img = cv2.imread('people.jpg')
 
cascade_frontface = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY )
faces = cascade_frontface.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
for face in faces:
    x,y,w,h = face
    print(face)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
cv2.imwrite("face_rec.jpg",img)
cv2.imshow("img",img)
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
plt.imshow(img)
cv2.waitKey()
cv2.destroyAllWindows()