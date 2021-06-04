import cv2
import numpy as np


def split_matrix(img):
    y, x = img.shape
    res = []
    for i in range(4):
        res.append([])
        for j in range(4):
            if i != 4 and j != 4:
                res[i].append(img[y // 4 * j:  y // 4 * (j + 1) - 1, x // 4 * i:x // 4 * (i + 1) - 1])
    return res


def get_aussian_mask(size):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g


def illumination(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = get_aussian_mask(4)
    h = img.shape[0]
    w = img.shape[1]
    res = np.zeros((4,4))
    m_array = split_matrix(img)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            res[i,j]=mask[i,j] * m_array[i][j].mean()
    result = res.mean()
    return result
