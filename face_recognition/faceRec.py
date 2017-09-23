# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 00:45:53 2017

@author: Administrator
"""

import os
import sys
import cv2
import numpy as np

def normalize(X, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):        
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            print "subject_path:",subject_path
            for filename in os.listdir(subject_path):
                print "filename:",filename
                try:
                    if (filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path, filename)
                    print "filepath:",filepath
                    
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if (im is None):
                        print "image " + filepath + " is none" 
                    # resize to given size (if given)
                    if (sz is not None):
                        im = cv2.resize(im, sz)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError, (errno, strerror):
                    print "I/O error({0}): {1}".format(errno, strerror)
                except:
                    print "Unexpected error:", sys.exc_info()[0]
                    raise
            c = c+1
    return [X,y]

if __name__=="__main__":
    people ={0:'dad',1:'mom',2:'son'}
    cam = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    [X,yy]=read_images("./")
    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(X),np.asarray(yy))
    while(True):
        ret,frame = cam.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))
            roi = cv2.resize(gray[y:y+h,x:x+w],(200,200))
            param=model.predict(roi)
            cv2.putText(frame,people[param],(x,y),cv2.FONT_HERSHEY_COMPLEX,2,255)
            #print(param[0],param[1])
        cv2.imshow("face rec",frame)
        if cv2.waitKey(100)&0xff==ord('q'):
            break;
    cam.release()
    cv2.destroyAllWindows()    