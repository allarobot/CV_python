# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 12:30:30 2017

@author: Administrator
"""

import os
def walk_img(imgpath):
    out =[]
    for path,directory,filename in os.walk(imgpath):
        for fn in filename:
            fullpathfile = os.path.join(path,fn)
            fname,ftype = fn.split('.')
            out.append([fullpathfile,fname,ftype])
    return out
        
if __name__=="__main__":
    out =walk_img('son')
    print(out)
    