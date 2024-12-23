import cv2
import numpy as np

"""
Набор функций для определения апроксимации изображения формой мозаичного изображения.

Набор публичных функций:
- relaxation : выполнение релаксационного алгоритма по апроксимации изображения формой мозаичного изображения
"""

_levels =[]

def update_levels(new_levels):
    _levels = new_levels

def get_levels():
    return _levels


def _get_levels(img, threshold = 16, show = False):
    '''
    Поиск уровней квантования

    Параметры
    ----------
    args : (img: np.array), (threshold: int (base =  16))

        - img - исходное изображение
        - threshold - порог поиска яркостей (минимальное количество пикселей для включения яркости в итоговый набор)
    
    Returns
    -------
        - list<int>: итоговый набор яркостей
    '''
    quantization_levels = []  # Изначально уровней нет
    
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])

    # Проанализируем гистограмму, чтобы найти количество уровней квантования
    for i in range(256):
        if hist[i] > threshold:
            quantization_levels.append(i)

    if show: print(f'Уровни квантования: {len(quantization_levels)}')
    return quantization_levels

def _find_level(pixel):
    '''
    Усреднение изображения по набору яркостей.

    Функция выполняемая для каждого пикселя.

    Параметры
    ----------
    args : (pixel: int)

        - pixel - пиксель изображения
    
    Returns
    -------
        - int: итоговая яркость пикселя
    '''
    min_diff = 256
    min_index = -1
    for i, level in enumerate(_levels):
        diff = abs(pixel - level)
        if (diff < min_diff):
            min_diff = diff
            min_index = level
    return min_index

def _get_masks(img, src_levels):
    '''
    Получение масок для набора яркостей

    Параметры
    ----------
    args : (image: np.array), (src_levels: list<int>)

        - image - входное изображение
        - src_levels - набор уровней квантования (яркостей)

    Returns
    -------
        - list<np.array>: набор индикаторных функций (масок)
    '''
    colorized = np.vectorize(_find_level)(img).astype(np.uint8)
    masks = []
    for level in src_levels:
        c = img.copy()*0
        c[colorized==level] = 1
        c[colorized!=level] = 0
        masks.append(c)
    return masks


def approximation(g, masks):
    '''
    Апроксимация формой мозаичного изображения.

    Параметры
    ----------
    args : (image: np.array), (masks: list<np.array>)

        - image - входное изображение
        - masks - набор индикаторных функций(масок)

    Returns
    -------
        - np.array: форма мозаичного изображения
        - list<int>: набор яркостей
    '''

    result = g.copy()*0
    colors = []
    for mask in masks:
        color = np.mean(np.ma.array(g,mask=1-mask))
        colors.append(int(color))
        result = result + mask*color
    return result.astype(np.uint8), colors

def _nevayz(image, proj):
    return np.square(np.linalg.norm(image - proj))

def relaxation(src, threshold = 16, N=20, show= False):
    '''
    Релаксационный алгоритм.

    Параметры
    ----------
    args : (src: np.array), (N: int)

        - v - входное изображение
        - N - количество итераций

    Returns
    -------
        - list<np.array>: набор изображений для каждой итерации
        - list<np.array>: набор масок для каждой итерации
        - list<list<int>>: набор яркостей для каждой итерации
    '''
    global _levels
    
    _levels = _get_levels([src], threshold=threshold, show = show)

    image = src.copy()
    
    images, masks, errors = [], [] ,[]

    for i in range(N):
        #Определяем индикаторные функции для новых цветов
        masks = _get_masks(src, _levels)
        

        #Вычисляем проекцию и новые цвета
        image, _levels = approximation(src, masks)
        
        images.append(image)
        errors.append(_nevayz(src,image))
        
        if show: print('.',end='')# showOneLine(images,[str(x) for x in range(len(images))],dim = 5)
        
    if show: print()
    return images, masks, _levels