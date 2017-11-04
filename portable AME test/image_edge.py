# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 19:53:42 2017

@author: Administrator
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_roi(img):
    r,c,h,w = 0,0,0,0
    gray = cv2.medianBlur(img,3)
#cv2.imshow("gray",gray)
    circles1 = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,2,100,param1=160,param2=80,minRadius=65,maxRadius=75)
    min_dis,i,max_dis,j=0,0,0,0
    for k,circle in enumerate(circles1[0]):
        dis = circle[0]**2+circle[1]**2
        if min_dis==0 or dis<min_dis:
            min_dis,i=dis,k            
            
        if max_dis==0 or dis>max_dis:
            max_dis,j=dis,k
    r,c,h,w = circles1[0][i][0],circles1[0][i][1],circles1[0][j][0]\
    -circles1[0][i][0],circles1[0][j][1]-circles1[0][i][1]
    return int(r),int(c),int(h),int(w)

        
    

def img_normal(img_left):
    min_val = img_left.min()
    max_val = img_left.max()
    ratio = 255.0/(max_val-min_val)
    img_left = (img_left-min_val)*ratio
    img_left = np.uint8(img_left)
    return img_left

def skip_margin(mask):
    x,y = mask.shape
    mask[0,:]=0
    mask[x-1,:]=0
    mask[:,0]=0
    mask[:,y-1]=0
    return mask
def line_edge_filter(mask,angle,torlerance,out=None):
    r,c = mask.shape
    binary = np.zeros((r,c),dtype='uint8')
    binary[mask==-1] = 255
    color = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(np.uint8(binary),2,0.02,40,minLineLength=20,maxLineGap=50)
    print(lines)
    x1,y1,x2,y2 = 0,0,0,0
    for line in lines:
        x,y,xx,yy=line[0]
        cv2.line(color,(x,y),(xx,yy),(255,0,0))
        if abs((xx-x)*np.cos(angle)+(yy-y)*np.sin(angle))/np.sqrt((xx-x)**2+(yy-y)**2)>np.cos(torlerance):
            if x1+y1+x2+y2==0:
                if (xx-x)*np.cos(angle)+(yy-y)*np.sin(angle)>0:
                    x1,y1,x2,y2 = x,y,xx,yy
                else:
                    x1,y1,x2,y2 = xx,yy,x,y
            else:
                if (x1-x)*np.cos(angle)+(y1-y)*np.sin(angle)>0:
                    x1,y1 = x,y
                if (x1-xx)*np.cos(angle)+(y1-yy)*np.sin(angle)>0:
                    x1,y1 = xx,yy                
                if (x-x2)*np.cos(angle)+(y-y2)*np.sin(angle)>0:
                    x2,y2 = x,y
                if (xx-x2)*np.cos(angle)+(yy-y2)*np.sin(angle)>0:
                    x2,y2 = xx,yy
    if out: 
        cv2.imwrite(out,color)            
    return (x1,y1,x2,y2)

def find_edge_mask(gray,position): 
#    color_img = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)       
    x,y = gray.shape
    mask = np.zeros((x,y),dtype='int32')
    if position == 'l' or position == 'r':
        kernel = np.array([[-1,-1,0,1,1],
                    [-1,-1,0,1,1],
                    [-1,-1,0,1,1],
                    [-1,-1,0,1,1],
                    [-1,-1,0,1,1],
                    [-1,-1,0,1,1],
                    [-1,-1,0,1,1]])
        mask[:,0:2] = 1
        mask[:,y-2:] = 2
    elif position =='b':
        kernel = np.array([[-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1],
                    [0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],])
        mask[0:2,0:y/2-100] = 1
        mask[0:2,y/2+100:] = 2
        mask[x-2:] = 3
        mask[:,y/2-80:y/2+80]=3
    else:# direction == 'v':
        kernel = np.array([[-1,-1,-1,-1,-1,-1,-1],
                    [-1,-1,-1,-1,-1,-1,-1],
                    [0,0,0,0,0,0,0],
                    [1,1,1,1,1,1,1],
                    [1,1,1,1,1,1,1],])
        mask[0:10,:] = 1
        mask[x-2:] = 2
        
    ret,mask=cv2.connectedComponents(np.uint8(mask))    
    gray = cv2.filter2D(gray,cv2.CV_16SC1,kernel) 
    gray = img_normal(gray)
    color = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)   
    cv2.watershed(color,mask)
    mask = skip_margin(mask)
    #plt.imshow(mask)  
    return mask

img = cv2.imread('./Data and Image/Image_Question4.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect roi
roi=find_roi(gray)
x,y,w,h = roi
x,y,w,h=x-100,y-200,w+200,h+400
img = gray[y:y+h,x:x+w]
cv2.imwrite("img_roi.jpg",img)
# find cicle
color = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
gray = cv2.medianBlur(img,3)       
circles2 = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,2,100,param1=160,param2=80,minRadius=320,maxRadius=350)
for circle in circles2[0]:
    cv2.circle(color,(circle[0],circle[1]),circle[2],(0,255,0),3)  

#find line features
r_top,c_top = 0,160
r_bot,c_bot = 910,160 
r_left,c_left = 330,30 
r_right,c_right = 330,930

img_left = gray[r_left:r_left+320,c_left:c_left+80]
mask_edge = find_edge_mask(img_left,'l')
#color[r_left:r_left+320,c_left:c_left+80][mask_edge==-1]=[255,0,0]
point=line_edge_filter(mask_edge,np.pi/2,np.pi/90,'1.jpg')
color[r_left:r_left+320,c_left:c_left+80]=cv2.line(color[r_left:r_left+320,c_left:c_left+80],point[0:2],point[2:],(0,255,0))
#print(point)

img_right = gray[r_right:r_right+320,c_right:c_right+80]
mask_edge = find_edge_mask(img_right,'r')
#color[r_right:r_right+320,c_right:c_right+80][mask_edge==-1]=[255,0,0]
point=line_edge_filter(mask_edge,np.pi/2,np.pi/50,'2.jpg')
color[r_right:r_right+320,c_right:c_right+80]=cv2.line(color[r_right:r_right+320,c_right:c_right+80],point[0:2],point[2:],(0,255,0))
print(point)

img_top = gray[r_top:r_top+80,c_top:c_top+700]
mask_edge = find_edge_mask(img_top,'t')
#color[r_top:r_top+70,c_top:c_top+700][mask_edge==-1]=[255,0,0]
point=line_edge_filter(mask_edge,0,np.pi/90,'3.jpg')
color[r_top:r_top+80,c_top:c_top+700]=cv2.line(color[r_top:r_top+80,c_top:c_top+700],point[0:2],point[2:],(0,255,0))
#print(point)

img_bot = gray[r_bot:,c_bot:c_bot+720]
mask_edge = find_edge_mask(img_bot,'b')
#color[r_bot:,c_bot:c_bot+720][mask_edge==-1]=[255,0,0]
point=line_edge_filter(mask_edge,0,np.pi/90,'4.jpg')
color[r_bot:,c_bot:c_bot+720]=cv2.line(color[r_bot:,c_bot:c_bot+720],point[0:2],point[2:],(0,255,0))
#print(point)

cv2.rectangle(color,(c_top,r_top),(c_top+700,r_top+80),(0,0,255))
cv2.rectangle(color,(c_bot,r_bot),(c_bot+720,r_bot+80),(0,0,255))
cv2.rectangle(color,(c_left,r_left),(c_left+80,r_left+320),(0,0,255))
cv2.rectangle(color,(c_right,r_right),(c_right+80,r_right+320),(0,0,255))    



cv2.imwrite("color_edge.jpg",color)
cv2.imshow("color",color)
cv2.waitKey()
cv2.destroyAllWindows()