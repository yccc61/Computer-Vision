import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt
from matchPics import matchPics


#Q3.5
#Read the image and convert to grayscale, if necessary

img1=cv2.imread("/Users/yunchuchen/Desktop/16385/assgn2/data/cv_cover.jpg")
matches_count=[]
for i in range(36):
	#Rotate Image
	rotated_cover=scipy.ndimage.rotate(img1, i*10, reshape=True)
	#Compute features, descriptors and Match features
	(matches, locs3, locs4)=matchPics(img1, rotated_cover)
	#Update histogram
	matches_count.append(len(matches))




plt.xlabel("Rotation in degrees")
plt.ylabel("Matches")
plt.title("Matches with degrees of rotation")
degrees=[i for i in range(0,360,10)]
plt.bar(degrees, matches_count, width=0.8)
#Display histogram

plt.show()
