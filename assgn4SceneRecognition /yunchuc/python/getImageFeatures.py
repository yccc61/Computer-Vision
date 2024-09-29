import numpy as np


def get_image_features(wordMap, dictionarySize):

    wordMap_flatten=wordMap.flatten()
    h=np.bincount(wordMap_flatten,minlength=dictionarySize)
    #L1 normalization
    h=h/np.sum(h)
    return h
