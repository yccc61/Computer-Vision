import numpy as np
import cv2
import skimage.io 
import skimage.color
#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH


cv_cover=cv2.imread("../data/cv_cover.jpg")
cv_desk=cv2.imread("../data/cv_desk.png")
hp_cover=cv2.imread("../data/hp_cover.jpg")

matches, locs1, locs2=matchPics(cv_cover, cv_desk)

#filter[:,0] contains indexes for matched coordinates in locs1
#filter out matched locs1 and locs2
cover_locs=locs1[matches[:,0]]
desk_locs=locs2[matches[:,1]]
#swap column and row for locs1 and locs2
tmp=np.copy(cover_locs[:,0])
cover_locs[:,0]=cover_locs[:,1]
cover_locs[:,1]=tmp

tmp=np.copy(desk_locs[:,0])
desk_locs[:,0]=desk_locs[:,1]
desk_locs[:,1]=tmp

#start matching, matching cover_locs to desk_locs
bestH2to1, inliers=computeH_ransac(desk_locs,cover_locs)

#resize hp_cover
cover_height=np.shape(cv_cover)[0]
cover_width=np.shape(cv_cover)[1]
hp_cover=cv2.resize(hp_cover, (cover_width, cover_height))


#composite H to fits the location
compositeResult=compositeH(bestH2to1,cv_desk,hp_cover)

cv2.imwrite("HarryPotterize.jpg", compositeResult)


#Write script for Q3.9
