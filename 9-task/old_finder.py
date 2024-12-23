import cv2
import numpy as np
from itertools import product
import noise_gen as ns
"""
Набор функций для поиска цифры по ее изображению из задания 4,5.

Набор публичных функций:
- findDigit : функция поиска цифры по ее изображению
"""

def _projection(g,f):
    p = g.copy()
    p[f==0] = np.ma.array(g,mask=f).mean()
    p[f==255] = np.ma.array(g,mask=(255-f)).mean()
    return p

def _projection2(g,f):
    p = g.copy()
    colors = []
    color1 = np.ma.array(g,mask=f).mean() #цвет цифры
    color2 = np.ma.array(g,mask=(255-f)).mean() #цвет фона

    if (color2>color1):
        p[f==0] = color1
        p[f==255] = color2
    else:
        p[::] =  g.mean()
        
    return p

def findDigit(image, samples, older=False):
    '''
    Функция поиска цифры по ее изображению

    Параметры
    ----------
    args : (image: np.array), (samples: List<np.array> ), (older: bool (base = False))

        - image - исходное изображение
        - samples - Набор бинаризованных форм цифр
        - older - использовать алгорим, не учитывающий упорядоченность яркостей
    
    Returns
    -------
        - int: определенная цифра
        - list<np.array>: проекции форм цифр на картинку
        - list<float>: разница между цифрами и изображением
    '''
    proj = _projection if older else _projection2
    min_diff = float('inf')
    E = np.ones(image.shape, dtype=np.uint8) * np.mean(image).astype(np.uint8)
    result = 0
    imgs, diffs = [], []
    for i, sample   in enumerate(samples):
        p = proj(image,sample)
        diff1 = cv2.absdiff(p,image)
        diff2 = cv2.absdiff(p,E)
        diff =  np.square(np.linalg.norm(diff1))/np.square(np.linalg.norm(diff2))
        if diff < min_diff:
            result = i
            min_diff = diff
        imgs.append(p)
        diffs.append(diff)
    
    return result, imgs, diffs


def get_error(sigma, samples, length = 100):
    # формируем словарь ошибок
    errors_map = {}
    for nums in product(range(10),repeat = 2):
        key = "-".join([str(x) for x in nums])
        errors_map[key] = 0
    
    #по каждой цифре
    for digit, dig_img in enumerate(samples):
        # выборка 100 для каждой цифры
        for tmp_img in ns.genNoises(dig_img, sigma, length = length):
            # если ошиблись, считаем ошибку
            result = findDigit(tmp_img, samples)[0]
            name = f"{digit}-{result}"        
            if digit != result:
                errors_map[name]+=1

    return errors_map