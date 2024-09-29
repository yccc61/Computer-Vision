import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
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

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(It.shape[1], x2)
    y2 = min(It.shape[0], y2)

    #create x and y coordinates based on dimensions
    y=np.arange(0,np.shape(It)[0])
    x=np.arange(0,np.shape(It)[1])
    #create spline representation for It and It1
    spline_I=RectBivariateSpline(y, x, It1)
    spline_T=RectBivariateSpline(y, x, It)
    rectX=np.arange(x1, x2)
    rectY=np.arange(y1, y2)
    meshx,meshy= np.meshgrid(rectX, rectY)

    #We could pre compute Jacobian
    #Note that according to the formula, should be Jacobian of T
    Tx=spline_T.ev(meshy,meshx, dy=1).flatten()
    Ty=spline_T.ev(meshy, meshx, dx=1).flatten()
    y=meshy.flatten()
    x=meshx.flatten()
    Jacobian= np.vstack((x*Tx, y*Tx, Tx, x*Ty, y*Ty, Ty)).T
    T_sub=spline_T(rectY, rectX)

    for i in range(maxIters):

        #similar wrap function, affine+translation
        rectX_wrap=meshx*(1+p[0])+meshy*p[1]+p[2]
        rectY_wrap=meshx*p[3]+meshy*(1+p[4])+p[5]


        I_sub=spline_I.ev(rectY_wrap, rectX_wrap)        
        b=(I_sub-T_sub).flatten()
        #solve the linear system without Hessian
        deltaP=np.linalg.lstsq(Jacobian,b)[0]

        #updating rule: W(x:p)<--W(x:p) o W(x:deltaP)^-1

        P=np.array([[1.0+p[0], p[1], p[2]],
                    [p[3],1.0+p[4],p[5]],
                    [0.0,0.0,1.0]])
        #convert p into matrix format for calculation
        DELTP=[[1.0+deltaP[0], deltaP[1], deltaP[2]],
               [deltaP[3],1.0+deltaP[4],deltaP[5]],
               [0.0,0.0,1.0]] 
        DELTP=np.array(DELTP)
        p=P@np.linalg.inv(DELTP)
        #only need first 6 parameters
        p-=np.array([[1,0,0],[0,1,0],[0,0,1]])
        p=p[0:2,:].flatten()
        if(np.linalg.norm(deltaP)<threshold):
            break



    # reshape the output affine matrix
    M = np.array([[1.0+p[0], p[1],    p[2]],
                 [p[3],     1.0+p[4], p[5]]]).reshape(2, 3)

    return M
