import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/Archive/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        

        filter_responses=extract_filter_responses(image, filterBank)
        if(method=="Random"):
            result_points=get_random_points(image, alpha)
        elif(method=="Harris"):
            result_points=get_harris_points(image, alpha, 0.05)
        # print(np.shape(image))
        # print(max(result_points[:,1]))
        # print(max(result_points[:,0]))

        #we are picking index pair in result_points for any "layer" of the responses.
        pixelResponses[alpha*i:alpha*(i+1)] = filter_responses[result_points[:,0], result_points[:,1]]
        
    dictionary = KMeans(n_clusters=K, random_state=0, algorithm='elkan').fit(pixelResponses).cluster_centers_
    return dictionary
