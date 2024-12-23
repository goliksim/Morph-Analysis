import numpy as np
import noise_gen as ns
from itertools import product

samples_bin = []
fields = []
fields_without_number = {}
array_mask = {}
uniq_masks = {}

def init(samples):
    global samples_bin
    samples_bin = samples
    h, w = samples_bin[0].shape

    fields = np.empty((h, w), dtype=str)
    fields = fields.astype('<U11')
    for number, img in enumerate(samples_bin):
        for i in range(h):
            for j in range(w):        
                if img[i, j] == 255:
                    fields[i, j]+=str(number)
    # Уникальные комбинации цифр, существубщие для наших картинок
    uniq_fields = [f for f in np.unique(fields) if len(f) > 0]
    # Маски для каждой уникальной комбинации
    
    for field in np.unique(fields):
        uniq_masks[field]= np.where(fields==field, 0, 255)
    # Набор уникальных комбинаций без определенной цифры
    
    for i in range(10):
        fields_without_number[i] = [x for x in uniq_fields if str(i) not in x]
    # Numpy True/False маски для каждого уникального набора цифр
    
    for field in uniq_fields:
        array_mask[field] = (fields == field)
    
def _Q(number, g):
    result = np.zeros(g.shape)
    for field in fields_without_number[number]:
        result[array_mask[field]] = np.ma.array(g, mask=uniq_masks[field]).mean()
    mask = np.logical_or(fields=='' , samples_bin[number]==255)
    result[mask] =  np.ma.array(g, mask=mask).mean()  
    return result

def _findDigit(g):
    min_diff = float('inf')
    result = 0
    imgs,diffs = [],[]
    
    for i in range(10):
        proj_img =  _Q(i, g)
        diff =  np.square(np.linalg.norm(proj_img))
       
        if diff < min_diff:
            result = i
            min_diff = diff
        imgs.append(proj_img)
        diffs.append(diff)
    
    return result, imgs, diffs


def get_error(sigma, samples, length = 100):
    #print('.',end='') 
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
            result = _findDigit(tmp_img)[0]
            name = f"{digit}-{result}"        
            if digit != result:
                errors_map[name]+=1
    return errors_map