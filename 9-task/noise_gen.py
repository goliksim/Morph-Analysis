import numpy as np
"""
Набор функций для генерации шума на изображении.

Набор публичных функций:
- genNoises : генерация выборки изображений с шумом.
"""

def addNoise(img, abs_sigma):
    noise = np.random.choice([-abs_sigma, abs_sigma], img.shape) ###!!!!!!!!!!!!!!!!!!!!!!!!!!
    return np.clip(img +  noise, 0, 255).astype(np.uint8)


def genNoises(img, abs_sigma, length = 100):
    '''
    Генерация выборки изображений с шумом.

    - Для генерации шума используется дискретное равномерное распределение сдвумя равновероятными значениями отклонений, σ и −σ.
    - Выборку генерируем так, чтобы в ней было одинаковое число изображений каждой цифры, например, по 100 изображений каждой цифры.

    Параметры
    ----------
    args : (img: np.array), (abs_sigma: int ), (length: int (base = 100))

        - img - исходное изображение
        - abs_sigma - Значение отклонения
    
    Returns
    -------
        - list<np.array>: итоговая выборка
    '''

    noises = []
    for i in range(length):
        noises.append( addNoise(img,abs_sigma))
    return noises