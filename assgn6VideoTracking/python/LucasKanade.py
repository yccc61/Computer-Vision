import numpy as np

from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1,y1,x2,y2 = rect

    #create x and y coordinates based on dimensions
    y=np.arange(0,np.shape(It)[0])
    x=np.arange(0,np.shape(It)[1])
    #create spline representation for It and It1
    spline_I=RectBivariateSpline(y, x, It1)
    spline_T=RectBivariateSpline(y, x, It)
    
    rectX=np.arange(x1, x2)
    rectY=np.arange(y1, y2)
    for i in range(maxIters):

        #Because of translation, wrap image is just simply addition
        #So to access the location of target after transformation, we only need (rectY_wrap, rectX_wrap)
        rectX_wrap=rectX+p[0]
        rectY_wrap=rectY+p[1]

        #Note that locations of rectangle may not be integer, and spline model is continuous, smooth surface, so we use spline in this case
        #spline_I.ev(y, x) y, x must be matrix
        #while spline_I(y,x) y,x should be array
        I_sub=spline_I(rectY_wrap, rectX_wrap)
        T_sub=spline_T(rectY, rectX)
        b=(T_sub-I_sub).flatten()

        #following the formula to calculate Jacobian, except no need for partialW/partial p
        Ix=spline_I(rectY_wrap, rectX_wrap, dx=1).flatten()
        Iy=spline_I(rectY_wrap, rectX_wrap, dy=1).flatten()
        Jacobian=np.transpose(np.vstack((Iy, Ix)))
        
        #solve the linear system without hessian
        deltaP=np.linalg.lstsq(Jacobian,b)[0]
        p=p+deltaP
        if(np.linalg.norm(deltaP)<threshold):
            break

    return p


