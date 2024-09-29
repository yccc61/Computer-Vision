import numpy as np

from scipy import signal    # For signal.gaussian function
from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    hsize=2*np.ceil(3*sigma)+1
    
    gaussian = np.array(signal.gaussian(hsize,sigma))
    filter= np.outer(gaussian,gaussian)
    filter= filter/np.sum(filter)
    smooth_img = myImageFilter(img0,filter)

    sobel_x=np.array([[-1,0,1],
                      [-2,0,2],
                      [-1,0,1]])
    sobel_y=np.array([[1,2,1],
                      [0,0,0],
                      [-1,-2,-1]])
    imgx=myImageFilter(smooth_img, sobel_x)
    imgy=myImageFilter(smooth_img, sobel_y)

 
    angles=np.degrees(np.arctan2(imgy,imgx))
    
    angles[angles<0]+=180
     
    angles = 45*np.round(angles/45)
    angles[angles==180]-=180
                    
    
    (img_height, img_width)=np.shape(img0)

    result=np.zeros_like(img0)

    magnitude=np.sqrt(imgx**2+imgy**2)
    result=np.sqrt(imgx**2+imgy**2)
    for i in range(1,img_height-1):
        for j in range(1,img_width-1):
            gradient_direction=angles[i,j]
            if(gradient_direction==0):
                neighboring_pixels=(magnitude[i, j+1], magnitude[i, j-1])
            elif(gradient_direction==45):
                neighboring_pixels=(magnitude[i-1, j+1], magnitude[i+1, j-1])
            elif(gradient_direction==90):
                neighboring_pixels=(magnitude[i+1, j], magnitude[i-1, j])
            else:
                neighboring_pixels=(magnitude[i-1, j-1], magnitude[i+1, j+1])
            if( magnitude[i,j]<neighboring_pixels[0] or magnitude[i,j]<neighboring_pixels[1]):
                result[i,j]=0

    return result




