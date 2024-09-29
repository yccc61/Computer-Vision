import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    

    M=np.sqrt(np.shape(img_threshold)[0]**2+np.shape(img_threshold)[1]**2)
    
    rhoScale=np.arange(0, M ,rhoRes)
    thetaScale=np.arange(0, 2*np.pi, thetaRes)

    img_hough=np.zeros((len(rhoScale), len(thetaScale)), dtype=int)
    
    i_s, j_s =np.nonzero(img_threshold)
    zipped=zip(i_s, j_s)
    for i, j in zipped:
        for k in range (len(thetaScale)):
            rho=np.cos(thetaScale[k])*j + np.sin(thetaScale[k])*i
            if(rho>0):
                rho_index=int(rho/rhoRes)
                img_hough[rho_index, k]+=1

    return img_hough, rhoScale, thetaScale

