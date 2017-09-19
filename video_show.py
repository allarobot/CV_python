# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy

imageCapture = cv2.VideoCapture(0)
success,frame = imageCapture.read()
size =(int(imageCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
       int(imageCapture.get(cv2.CAP_PROP_FRAME_WIDTH)))
while success and cv2.waitKey(1) == -1:
    cv2.imshow("LIVE",frame)
    success,frame = imageCapture.read()
    
cv2.destroyWindow("LIVE")    
imageCapture.release()
