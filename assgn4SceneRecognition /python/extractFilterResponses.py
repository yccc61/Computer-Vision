import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *

from createFilterBank import create_filterbank


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))
    I=rgb2lab(I)
    filterResponses=np.zeros((np.shape(I)[0], np.shape(I)[1], 60))
    #for every filter and every channel, compute the corresponding responses
    for j in  range(len(filterBank)):
        for i in range(3):
            I_channel=I[:,:,i]
            channel_tmp=imfilter(I_channel, filterBank[j])
            filterResponses[:,:,j*3+i]=channel_tmp
    return filterResponses

# img_1=cv.imread("../data/Archive/desert/sun_adpbjcrpyetqykvt.jpg")
# filterBank=create_filterbank()
# result=extract_filter_responses(cv.cvtColor(img_1, cv.COLOR_BGR2RGB),filterBank)
# for i in range(20):
#     response=result[:,:,i*3:(i+1)*3]
#     response=255*(response-response.min())/response.ptp()
#     response=cv.cvtColor(response.astype(np.uint8), cv.COLOR_RGB2GRAY)
#     cv.imwrite(f"{i}.jpg", response)
#     print("Done")



