import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   M: the Affine warp matrix [2x3 numpy array]

    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros((6,1))
    p=p.flatten()
    x1,y1,x2,y2 = rect


    #bound the size
    x1=max(0, x1)
    y1=max(0, y1)
    x2=min(x2,It.shape[1])
    y2=min(y2,It.shape[0])

    #create x and y coordinates based on dimensions
    y=np.arange(0,np.shape(It)[0])
    x=np.arange(0,np.shape(It)[1])
    #create spline representation for It and It1
    spline_I=RectBivariateSpline(y, x, It1)
    spline_T=RectBivariateSpline(y, x, It)
    
    rectX=np.arange(x1, x2)
    rectY=np.arange(y1, y2)
    meshx, meshy = np.meshgrid(rectX, rectY)
    
    for i in range(maxIters):

        #wrap matrix is 
        #[1+p0, p1, p2
        # p3,1+p4, p5
        # 0,0,1]
        rectX_wrap=meshx*(1+p[0])+meshy*p[1]+p[2]
        rectY_wrap=meshx*p[3]+meshy*(1+p[4])+p[5]
        #Note that locations of rectangle may not be integer, and spline model is continuous, smooth surface, so we use spline in this case
        #spline_I.ev(y, x) y, x must be matrix
        #while spline_I(y,x) y,x should be array
        I_sub=spline_I.ev(rectY_wrap, rectX_wrap)
        T_sub=spline_T(rectY, rectX)
        b=(T_sub-I_sub).flatten()

        #following the formula to calculate Jacobian, its different from handout because of the arrangement of parameters
        #[\partial u is dy, while \partial v is dx]
        Ix=spline_I.ev(rectY_wrap, rectX_wrap, dy=1).flatten()
        Iy=spline_I.ev(rectY_wrap, rectX_wrap, dx=1).flatten()
        y=rectY_wrap.flatten()
        x=rectX_wrap.flatten()
        Jacobian= np.vstack((x*Ix, y*Ix, Ix, x*Iy, y*Iy, Iy)).T
        
        #solve the linear system without Hessian
        deltaP=np.linalg.lstsq(Jacobian,b)[0]
        p=p+deltaP
        if(np.linalg.norm(deltaP)<threshold):
            break


    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
