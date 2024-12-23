import cv2
import matplotlib.pyplot as plt

"""
Набор вспомогательных функций для заданий морфологического анализа.

Набор публичных функций:
- showOneLine : удобное отображение множества изображений
"""

def showOneLine(imgs, texts,size=(18,12), dim = 3):
    '''
    Удобное отображение множества изображенийПоиск уровней квантования

    Параметры
    ----------
    args : (imgs: list<np.array>), (texts: list<String>), (size: tuple (base =  (28, 16))) , (dim: int (base =  3))

        - imgs - набор изображений
        - texts - набор подписей
        - size - размер отображения
        - dim - количество изображений по горизонтали
    
    Returns
    -------
        list
    '''
    plt.figure(figsize=size)
    length = len(imgs)
    for i, (image, text) in enumerate(zip(imgs, texts)):
        plt.subplot(length//dim+1, dim, i+1)
        plt.title(text)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()