import cv2
import numpy as np
import mosaic_approx as ma

def color_var(window, masks):
    
    #Берем обратный цвет центра и закрашиваем предельным им
    main_color = 255- np.mean(window[masks[0]==1])
    if main_color > 128: main_color = 255
    else: main_color=0

    for mask in masks[:-1]:
        window[mask==1] = main_color   
    
    return window
    
def clear_window(window, img_masks, thr):
    E = np.ones(window.shape, dtype=np.uint8) * np.mean(window).astype(np.uint8)
    p, _ = ma.approximation(window,img_masks)
    diff1 = cv2.absdiff(p,window)
    diff2 = cv2.absdiff(p,E)
    
    diff =  np.square(np.linalg.norm(diff1))/np.square(np.linalg.norm(diff2))
    if diff < thr:
        window = color_var(window, img_masks)
    return window

def moving_window(img, action, *args, w_size= (3, 3), ):
    s, k = w_size
    n, m = img.shape
    image = img.copy()

    # Создим новое изображение белого цвета с большими размерами
    n, m = n + 2, m + 2
    extended_image = (np.ones((n, m), dtype=np.uint8)*255).astype(np.uint8)
    # Вставьте исходное изображение в центре расширенного изображения
    extended_image[1: n-1, 1: m-1] = image

    # Пройдемся по изображению с помощью скользящего окна
    for i in range(0, n - s + 1):
        for j in range(0, m - k + 1):
            # Выделим текущее окно
            window = extended_image[i:i + s, j:j + k]
            extended_image[i:i + s, j:j + k] = action(window, *args)

    return extended_image[1: n-1, 1: m-1]

def sp_clear(img, threshold=1.43):
    sp_img = cv2.cvtColor(cv2.imread('sp_img.png'), cv2.COLOR_BGR2GRAY)
    sp_f, sp_masks, sp_levels  = ma.relaxation(sp_img, threshold=0, N=1)

    return  moving_window(img, clear_window,sp_masks,threshold)


def crc_clear(img, threshold=0.89, w_size = (5,5)):
    crc_img = cv2.cvtColor(cv2.imread('crc_img.png'), cv2.COLOR_BGR2GRAY)
    crc_f, crc_masks, crc_levels  = ma.relaxation(crc_img, threshold=0, N=1)

    return  moving_window(img, clear_window,crc_masks,threshold, w_size= w_size)