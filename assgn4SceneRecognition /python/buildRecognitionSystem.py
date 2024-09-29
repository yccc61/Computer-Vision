import numpy as np
import pickle
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
import cv2 as cv

with open("../data/traintest.pkl", "rb") as file:
    meta=pickle.load(file)
train_names=meta["train_imagenames"]
train_labels=meta["train_labels"].reshape(-1,1)

filterBank=create_filterbank()
dictionaries=["Random", "Harris"]
for dictionary_name in dictionaries:
    with open(f"dictionary{dictionary_name}.pkl", "rb") as file:
        curr_dictionary=pickle.load(file)
    train_features=[]
    for (i, img_path) in enumerate(train_names):
        wordMap_path=f"../data/Archive/{img_path[:-4]}_{dictionary_name}.pkl"
        with open(wordMap_path, "rb") as file:
            wordMap=pickle.load(file)
        h=get_image_features(wordMap, np.shape(curr_dictionary)[0])
        train_features.append(h)
    train_features=np.array(train_features)
    classifier={
        "dictionary":curr_dictionary,
        "filterBank":filterBank,
        "trainFeatures":train_features,
        "trainLabels": train_labels
    }
    with open(f'vision{dictionary_name}.pkl', 'wb') as fh:
        pickle.dump(classifier, fh)

