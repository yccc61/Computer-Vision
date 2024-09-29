import numpy as np
import cv2 


def myHoughLines(img_hough, nLines):
    
    img_hough_uint8=img_hough.astype(np.uint8)
    kernel_neighbors=np.ones((3, 3), dtype=np.uint8)
    dilated = cv2.dilate(img_hough_uint8, kernel_neighbors)
    maximum_mask=(dilated==img_hough)
    nms_result=np.multiply(maximum_mask, img_hough)
    (rhos, thetas)=np.nonzero(nms_result)
    def sortFunction(coord):
        i,j=coord
        return -img_hough[i,j]
    coordinates=list(zip(rhos, thetas))
    sorted_coordinates = sorted(coordinates, key=sortFunction)
    rhos, thetas= list(zip(*sorted_coordinates))
    return rhos[0:nLines], thetas[0:nLines]


    
