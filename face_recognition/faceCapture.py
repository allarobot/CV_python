# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:13:26 2017

@author: Administrator
"""
import cv2
#import matplotlib.pyplot as plt
def face_capture(folder='.'):
    videoCapture = cv2.VideoCapture(0)
    cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    i=0
    while True:
        ret,frame = videoCapture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade_face.detectMultiScale(gray,1.3,5)   
        for face in faces:
            x,y,w,h = face
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255))
            img_face=cv2.resize(gray[y:y+h,x:x+w],(200,200))
            cv2.imwrite("%s/%d.jpg"%(folder,i),img_face)
            i += 1
            print(i)
    
        cv2.imshow("face recog",frame)
        if cv2.waitKey(1000/12)&0xff ==ord('q'):
            break;
    videoCapture.release()    
    cv2.destroyAllWindows()
if __name__=="__main__":
    face_capture(".")