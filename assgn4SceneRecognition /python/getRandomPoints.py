import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def get_random_points(I, alpha):
    #row first, then column
    random_column=np.random.randint(0, np.shape(I)[1], size=alpha)
    random_row=np.random.randint(0, np.shape(I)[0], size=alpha)
    return np.column_stack((random_row, random_column))


# sun_akgyyhdnnpenxrwv.jpg
# sun_aciggnzupbzygsaw.jpg
# sun_akgyyhdnnpenxrwv.jpg
# img_1=cv.imread("../data/Archive/campus/sun_aciggnzupbzygsaw.jpg")
# img_1=cv.cvtColor(img_1, cv.COLOR_BGR2RGB)

# result_points=get_random_points(img_1, 500)
# plt.imshow(img_1)
# x_coordinates=result_points[:,1]
# y_coordinates=result_points[:,0]
# plt.scatter(x_coordinates, y_coordinates, s=2.7, marker="o", c="orange")
# plt.savefig('Random_points3.jpg')
