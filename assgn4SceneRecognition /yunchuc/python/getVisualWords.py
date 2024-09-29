import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
import pickle
import cv2 as cv
from createFilterBank import create_filterbank
import skimage

def get_visual_words(I, dictionary, filterBank):

    #To classify every pixel, first apply every pixel to the 20 filter
    filterResponses=extract_filter_responses(I, filterBank)
    #filterResponses would be having height of (HxW) and width of 3*len(filterResponses)
    #Meaning every column corresponded to every image
    #In order words, every row corresponded to a pixel of the image with 3*len(filterReponses) transformation
    filterResponses=filterResponses.reshape(-1,3*len(filterBank))

    #dictionary has the size of K*(3*len(filterResponses))
    #Meaning there are K visual words. 
    #If we indentify the a word by its letter=>here we identify/characterize a visual word by 3*len(filterResponses) characteristics
    #This is because 3 colors*20filters => 60 characteristics  

    #Compute the euclidean distance between the characteristics of pixel and the visual word allow us to "count" the pixel
    #The distance will have the size of (hxw) * K, each row represent the distance of a pixel to K clusters
    distance=cdist(filterResponses,dictionary,"euclidean")
    #get the argmin of each row
    result=np.argmin(distance, axis=1)
    result=result.reshape(np.shape(I)[0], np.shape(I)[1])

    return result

#script for generating colored map

# with open("dictionaryRandom.pkl", "rb") as file1:
#     dictionary_random=pickle.load(file1)
# with open("dictionaryHarris.pkl", "rb") as file2:
#     dictionary_harris=pickle.load(file2)
# filterBank=create_filterbank()
# # sun_akgyyhdnnpenxrwv.jpg
# # sun_aciggnzupbzygsaw.jpg
# # sun_akgyyhdnnpenxrwv.jpg
# image_paths=["../data/Archive/airport/sun_aesovualhburmfhn.jpg",
#              "../data/Archive/airport/sun_aesyuxjawitlduic.jpg",
#              "../data/Archive/campus/sun_aciggnzupbzygsaw.jpg",
#              "../data/Archive/campus/sun_akgyyhdnnpenxrwv.jpg",
#              "../data/Archive/campus/sun_abpxvcuxhqldcvln.jpg"]
# for i, img_path in enumerate(image_paths):
#     curr_img=cv.imread(img_path)
   
#     # curr_img=cv.cvtColor(curr_img, cv.COLOR_BGR2RGB)
#     harris_map=get_visual_words(curr_img,dictionary_harris,filterBank)
#     harris_visual=skimage.color.label2rgb(harris_map)
#     harris_visual=(harris_visual-harris_visual.min())*225/(harris_visual.max()-harris_visual.min())
#     harris_visual=np.array(harris_visual, np.float64)
#     print("success")
#     # cv.imshow("harris_visual", harris_visual)
#     # cv.waitKey(1000)

#     cv.imwrite(f"harris{i}.jpg", harris_visual)

#     random_map=get_visual_words(curr_img, dictionary_random, filterBank)
#     random_visual=skimage.color.label2rgb(random_map)
#     #normalize the color
#     random_visual=(random_visual-random_visual.min())*255/(random_visual.max()-random_visual.min())
#     random_visual=np.array(random_visual, np.float64)
#     cv.imwrite(f"random{i}.jpg", random_visual)
#     # cv.imshow("random_visual", random_visual)
#     # cv.waitKey(0)

#     # cv.destroyAllWindows()
   




