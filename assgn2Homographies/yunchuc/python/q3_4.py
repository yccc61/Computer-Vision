import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
import scipy
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')


rotated_cover=scipy.ndimage.rotate(cv_cover, 10, reshape=True)

matches, locs1, locs2 = matchPics(cv_cover, rotated_cover)


#display matched features
plotMatches(cv_cover, rotated_cover, matches, locs1, locs2)
