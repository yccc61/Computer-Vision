import numpy as np
from utils import chi2dist
from scipy.spatial.distance import cdist

def get_image_distance(hist1, histSet, method):
    '''
    Parameters:
    - hist1:  1 * k matrix of histogram
    - histSet: matrix of 1 * k matrix of histogram, and the target
    - method: how the distance is computed, either euclidean or chi2

    Returns
    - Vector of distances
    '''
    # print(hist1.shape)

    if method == "euclidean":
        dist=np.zeros(np.shape(histSet)[0])
        for i in range(len(dist)):
            dist[i]=cdist(hist1.reshape(1,-1), histSet[i,:].reshape(1,-1),metric='euclidean')
    elif method== "chi2":
        def chiDistance(hist):
            return chi2dist(hist1, hist)
        dist=np.apply_along_axis(chiDistance, 1  ,histSet)

    return dist
