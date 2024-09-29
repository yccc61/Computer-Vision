import numpy as np
from scipy import signal 

def myImageFilter(img0, h):
    

    (height, width)=np.shape(img0)
    filHeight,filWidth=np.shape(h)

    # import pdb;pdb.set_trace()
    pad_width=((filHeight//2, filHeight//2),(filWidth//2, filWidth//2))
    padded_img0=np.pad(img0, pad_width,'edge')
    result_image=np.zeros_like(img0)

    for i in range (0, height, 1):
        for j in range (0, width, 1):
            result_image[i, j]=np.sum(h * padded_img0[i:(filHeight+i), j:(filWidth+j)])
    return result_image



