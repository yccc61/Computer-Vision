import numpy as np
import cv2
#Import necessary functions
import skimage
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac
from planarH import compositeH


cv_cover=cv2.imread("../data/cv_cover.jpg")

k=4
source_book = np.load("../data/book.npy")
source_movie = np.load("../data/ar_source.npy")

frame_width = 640
frame_height = 480
out = cv2.VideoWriter('../result/ar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# #Write script for Q4.1

min_frame_count=min(np.shape(source_book)[0], np.shape(source_movie)[0])

for itr in range(min_frame_count):
    matches, locs1, locs2=matchPics(cv_cover, source_book[itr])
    #filter[:,0] contains indexes for matched coordinates in locs1
    #filter out matched locs1 and locs2

    numberMatches=np.shape(matches)[0]
    if(numberMatches>=k):
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
        bestH2to1, inliers=computeH_ransac(desk_locs,cover_locs, LoopTimes=400)    

    #resize hp_cover
    cover_height=np.shape(cv_cover)[0]
    cover_width=np.shape(cv_cover)[1]
    movie=cv2.resize(source_movie[itr], (cover_width, cover_height))

    compositeImage = compositeH(bestH2to1,source_book[itr],movie)
    compositeImage = cv2.convertScaleAbs(compositeImage) 

    #writing into output
    out.write(compositeImage)
# When everything done, release the video write object
out.release()




