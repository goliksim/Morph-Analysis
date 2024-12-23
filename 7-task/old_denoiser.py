import cv2
import numpy as np

def comboDilate(image, kernel_x,kernel_y):
    d_x = cv2.dilate(image, kernel_x, iterations=1)
    d_y = cv2.dilate(image, kernel_y, iterations=1)
    d_comb = ((d_y.astype(np.uint16) + d_x.astype(np.uint16)) /2).astype(np.uint8)
    return d_x, d_y, d_comb


def mseImages(image1, image2):
    difference_1 = cv2.subtract(image2, image1)/255
    difference_2 = cv2.subtract(image1, image2)/255

    squared_difference = np.square(difference_1)

    # Вычислите среднее квадратичное отклонение (MSE)
    return np.mean((np.square(difference_1)+ np.square(difference_2))/2, dtype= np.float64)