# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:30:30 2017

@author: Administrator
"""

import os
import cv2
def reindex(imgpath):
    for path,directory,filename in os.walk(imgpath):
        print("path:",path)
        print("directory:",directory)
        print("filename:",filename)
        for fn in filename:
            fullpathfile = os.path.join(path,fn)
            print("fullname:",fullpathfile)
            img = cv2.imread(fullpathfile)
            cv2.imshow("imgfile",img)
            if cv2.waitKey(100)&0xff == ord('q'):
                break;
    cv2.waitKey()
    cv2.destroyAllWindows()
        
if __name__=="__main__":
    reindex('son')
    