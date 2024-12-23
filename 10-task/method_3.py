import cv2
import numpy as np
import old_denoiser as old

def sp_clear(img):
    kernel_x = np.ones((1, 2), np.uint8) 
    kernel_y = np.ones((2, 1), np.uint8)
    sp_dilated_x, sp_dilated_y,sp_dilated_comb  = old.comboDilate(img, kernel_x, kernel_y)

    return sp_dilated_comb


def crc_clear(img):
    kernel_size_2 = (3, 3)

    crc_smoothed_image = cv2.blur(img, kernel_size_2)
    _, crc_smoothed_image_b = cv2.threshold(crc_smoothed_image, 128, 255, cv2.THRESH_BINARY)

    return  crc_smoothed_image_b