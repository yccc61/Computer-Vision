import pickle
import numpy as np
import cv2 as cv
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#importing files:

distance_methods=["euclidean", "chi2"]

with open("../data/traintest.pkl","rb") as file1:
    traintest=pickle.load(file1)
    test_imgs=traintest["test_imagenames"]
    test_labels=traintest["test_labels"]

with open("visionHarris.pkl", "rb") as file2:
    harris=pickle.load(file2)
    harris_dictionary=harris["dictionary"]
    harris_trainFeatures=harris["trainFeatures"]
    harris_trainLabels=harris["trainLabels"]

with open("visionRandom.pkl","rb") as file3:
    random=pickle.load(file3)
    random_dictionary=random["dictionary"]
    random_trainFeatures=random["trainFeatures"]
    random_trainLabels=random["trainLabels"]

for method in distance_methods:
    random_prediction=[]
    harris_prediction=[]
    for (i, img_path) in enumerate(test_imgs):
        random_path=f"../data/Archive/{img_path[0:-4]}_Random.pkl"
        harris_path=f"../data/Archive/{img_path[0:-4]}_Harris.pkl"
        
        # print(random_path)
        harris_wordmap=pickle.load(open(harris_path,"rb"))
        harris_histogram=get_image_features(harris_wordmap, len(harris_dictionary))
        harris_distance=get_image_distance(harris_histogram, harris_trainFeatures,method)
        
        random_wordmap=pickle.load(open(random_path,"rb"))
        random_histogram=get_image_features(random_wordmap,len(random_dictionary))
        random_distance=get_image_distance(random_histogram, random_trainFeatures, method)
        
        random_prediction+=[random_trainLabels[np.argmin(random_distance)]]
        harris_prediction+=[harris_trainLabels[np.argmin(harris_distance)]]
        

    random_accuracy=accuracy_score(test_labels, random_prediction)
    random_confusion=confusion_matrix(test_labels, np.array(random_prediction))
    harris_accuracy=accuracy_score(test_labels, harris_prediction)
    harris_confusion=confusion_matrix(test_labels, np.array(harris_prediction))

    print(f"Method:{method}")
    print(f"Harris: accuracy{harris_accuracy}")
    print(f"Harris: confusion")
    print(harris_confusion)
    print(f"Random: accuracy{random_accuracy}")
    print(random_confusion)


