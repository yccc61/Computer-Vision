import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation


from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import matplotlib.pyplot as plt
# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    sigma_est = skimage.restoration.estimate_sigma(image, channel_axis=-1, average_sigmas=True)
    denoised_img=skimage.restoration.denoise_bilateral(image, sigma_color=sigma_est,channel_axis=-1)
    gray_image = skimage.color.rgb2gray(denoised_img)
    thresh = threshold_otsu(gray_image)

    bw = closing(gray_image<thresh, skimage.morphology.square(7))
    
    label_image = label(bw,background=0,connectivity=2)
    for region in regionprops(label_image):
        if region.area >= 100:
            minr, minc, maxr, maxc = region.bbox
            bboxes.append([minr, minc, maxr, maxc])


    return bboxes, ~bw
