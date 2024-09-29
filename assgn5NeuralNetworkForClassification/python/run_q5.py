import os
import numpy as np
import matplotlib
# matplotlib.use('agg') 
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

import copy
import string
import pickle

from nn import *
from q5 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

def sortColumn(coordinates):
    #return minc
    return coordinates[1]

def cluster_bboxes(bboxes, threshold):
    clusters=[]
    curr_cluster=[bboxes[0]]
    result=[]
    for i in range(1,len(bboxes)):
        if abs(curr_cluster[-1][0]-bboxes[i][0])<=threshold:
            curr_cluster.append(bboxes[i])
        else:
            clusters.append(curr_cluster)
            curr_cluster=[bboxes[i]]
    clusters.append(curr_cluster)
    for cluster in clusters:
        result.append(sorted(cluster, key=sortColumn))
    return result


for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    plt.imshow(bw, cmap="gray")
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
    
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    bboxes=cluster_bboxes(bboxes, 100)

    output=[]
    for row in range(len(bboxes)):
        word=""
        for col in range(len(bboxes[row])):
            
            # crop the bounding boxes

            #Corp the image first, then padding-- but letter should be at the middle, 
            #So resize to 28 then pad to 32 so that there is some edges
            #Last, use dilation to make letter thicker..
            minr, minc, maxr, maxc=bboxes[row][col]
            # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
            centerC=minc+(maxc-minc)//2
            centerR=minr+(maxr-minr)//2
            
            diff=max(maxr-minr, maxc-minc)
            letter_img=bw[centerR-diff//2:centerR+diff//2,centerC-diff//2:centerC+diff//2]
            resize=skimage.transform.resize(letter_img, (28, 28))
            
            padded_letter=np.pad(resize,((2,2),(2,2)), mode='constant', constant_values=255)
            padded_letter=skimage.morphology.dilation(~padded_letter)
            padded_letter=~padded_letter
            # note.. before you flatten, transpose the image (that's how the dataset is!)
            padded_letter=np.transpose(padded_letter)
            x=padded_letter.reshape(1,1024)
            
            import pickle
            import string
            # load the weights
            letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
            # run the crops through your neural network and print them out
            params = pickle.load(open('q3_weights.pickle','rb'))
            h1=forward(x, params, name="layer1", activation=sigmoid)
            probs=forward(h1, params, name="output", activation=softmax)
            prediction=np.argmax(probs, axis=1)
            word+=str(letters[prediction][0])
        output.append(word)
    print(output)
   
   
   
    
    
    
    

