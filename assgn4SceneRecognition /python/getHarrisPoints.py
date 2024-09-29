import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter

from scipy import signal
import matplotlib.pyplot as plt

def imfilter_Harris(I, h):
    I_f = signal.correlate(I, h, mode='same')
    return I_f
def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0


    #window is 3x3, so pad 1
    Iy, Ix=np.gradient(I)
    Ixx=Ix*Ix
    Iyy=Iy*Iy
    Ixy=Ix*Iy

    sum_filter=np.ones((3,3))
    Sxx=imfilter_Harris(Ixx, sum_filter) 
    Syy=imfilter_Harris(Iyy, sum_filter)
    Sxy=imfilter_Harris(Ixy, sum_filter)


    det_covmat=Sxx*Syy-Sxy*Sxy
    trace=Sxx+Syy
    R=det_covmat-k*(trace**2)

    #First, convert the array into 1d array, and then sort arguments with descending order
    #Then convert it back using unravel_index, but need to use zip to zip the results back
    sorted_R=np.argsort(R.flatten())
    rows,cols=np.unravel_index(sorted_R[-alpha:],R.shape)
    points=[[x,y] for (x,y) in zip(rows,cols)]
    
    return np.array(points)

# sun_abslhphpiejdjmpz.jpg
# img_1=cv.imread("../data/Archive/campus/sun_aciggnzupbzygsaw.jpg")
# img_1=cv.cvtColor(img_1, cv.COLOR_BGR2RGB)

# result_points=get_harris_points(img_1, 500, 0.05)
# print(result_points)
# plt.imshow(img_1)
# print(result_points[:,0])
# x_coordinates=result_points[:,1]
# y_coordinates=result_points[:,0]
# plt.scatter(x_coordinates, y_coordinates, s=2.7, marker="o", c="orange")
# plt.savefig('harris3.jpg')

