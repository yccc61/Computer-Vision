import numpy as np
import pickle 
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

with open("../data/traintest.pkl","rb") as file1:
    traintest=pickle.load(file1)
    test_imgs=traintest["test_imagenames"]
    test_labels=traintest["test_labels"]

with open("visionRandom.pkl","rb") as file3:
    random=pickle.load(file3)
    random_dictionary=random["dictionary"]
    random_trainFeatures=random["trainFeatures"]
    random_trainLabels=random["trainLabels"]

chosen_method="Random"
chosen_distance="chi2"

accuracies=[]
confusion_matrices=[]
for k in range(1,41,1):
    dictionary=random_dictionary
    train_features=random_trainFeatures
    train_labels=random_trainLabels

    predictions=np.zeros(np.shape(test_labels)[0])
    for (i,img) in enumerate(test_imgs):
        wordMap=pickle.load(open(f"../data/Archive/{img[:-4]}_{chosen_method}.pkl", "rb"))
        img_feature=get_image_features(wordMap, len(dictionary))
        img_distance=get_image_distance(img_feature, train_features, chosen_distance).flatten()
        #Obtain the k nearest neighbors by sorting the distance
        k_nearest_neightbor=np.argsort(img_distance)
        k_nearest_neightbor=k_nearest_neightbor[:k]
        k_nearest_neighbor_labels=train_labels[k_nearest_neightbor].astype(int).flatten()
        predictions[i]= np.argmax(np.bincount(k_nearest_neighbor_labels))
    accuracies.append(accuracy_score(test_labels, predictions))
    confusion_matrices.append(confusion_matrix(test_labels, predictions, labels=np.arange(1,9)))

accuracies=np.array(accuracies)
confusion_matrices=np.array(confusion_matrices)

print(accuracies)
accuracy_best=np.max(accuracies)
k_best=np.argmax(accuracies)+1
confusion_best=confusion_matrices[np.argmax(accuracies)]
print(f"Best Accuracy: {accuracy_best}")
print(f"Best k: {k_best}")
print(f"Best confusion:")
print(confusion_best)

plt.plot([k for k in range(1,41)], accuracies)
plt.ylabel('Accuracy')
plt.xlabel('K')
plt.title('Accuracy with different k')
plt.savefig('plot.png')