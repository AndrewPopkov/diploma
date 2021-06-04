import numpy as np
from PIL import Image, ImageStat





def calc_hbs_brightness(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    count_canal = image_array.shape[2]
    norm_array= np.zeros(image_array.shape)
    for i in range(count_canal):
        norm_array[:, :, i]= image_array[:, :, i] / image_array[:, :, i].max()
    max_array= norm_array.max(axis=2)
    result = max_array.mean()
    return result
# 0.2053 0.7125 0.4670
# 1.8537 − 1.2797 −0.4429
# − 0.3655 1.0120 − 0.6104
def calc_bezryadinetal_brightness(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    count_canal = image_array.shape[2]
    h = image_array.shape[0]
    w = image_array.shape[1]
    norm_array= np.zeros(image_array.shape)
    m= np.array([[0.2053, 0.7125, 0.4670],
                 [1.8537, -1.2797, -0.4429],
                 [-0.3655, 1.0120, -0.6104]])
    for i in range(count_canal):
        norm_array[:, :, i]= image_array[:, :, i] / image_array[:, :, i].max()
    result = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            result[i,j] = np.linalg.norm(np.dot(m,norm_array[i,j,:]))
    result = result.mean()
    return result
