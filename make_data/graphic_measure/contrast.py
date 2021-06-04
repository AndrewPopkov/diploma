import numpy as np
from PIL import Image


def calc_rms_contrast(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    count_canal = image_array.shape[2]
    rms = []
    for i in range(count_canal):
        # mean=np.mean(image_array[:, :, i])
        normed = image_array[:, :, i] / image_array[:, :, i].max()
        rms.append(np.std(normed))
    result = np.array(rms).mean()
    return result

def calc_michelson_contrast(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    h = image_array.shape[0]
    w = image_array.shape[1]
    count_canal = image_array.shape[2]
    michelson = []
    for i in range(count_canal):
        michelson.append((image_array[:, :, i].max()-image_array[:, :, i].min())/(int(image_array[:, :, i].max())+int(image_array[:, :, i].min())))
    result = np.array(michelson).mean()
    return result
