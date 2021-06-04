import cv2
import numpy as np


def to_uint8(x):
    x = x.astype(np.float)
    x *= 255.0 / x.max()
    x = x.astype(np.uint8)
    return x


def derivatives_horizontal(img):
    img = img.astype(np.int32)
    der_array = np.gradient(img, axis=1)
    der_array.astype(np.uint8)
    return der_array


def derivatives_vertical(img):
    img = img.astype(np.int32)
    der_array = np.gradient(img, axis=0)
    der_array.astype(np.uint8)
    return der_array


def L1_norm(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = derivatives_horizontal(img)
    y = derivatives_vertical(img)
    x = derivatives_horizontal(x)
    y = derivatives_vertical(y)
    absX = cv2.convertScaleAbs(x)  # Перенести обратно на uint8
    absY = cv2.convertScaleAbs(y)
    s = absX + absY
    result = s.mean()
    return result


def energy_Laplacian(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = derivatives_horizontal(img)
    y = derivatives_vertical(img)
    x = derivatives_horizontal(x)
    y = derivatives_vertical(y)
    absX = cv2.convertScaleAbs(x)  # Перенести обратно на uint8
    absY = cv2.convertScaleAbs(y)
    s = absX + absY
    result = np.square(s).mean()
    return result


def kryszczuk_drygajlo_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_array = np.array(img)
    h = image_array.shape[0]
    w = image_array.shape[1]
    x_array = np.abs(img[:, :-1] + img[:, 1:])
    y_array = np.abs(img[:-1, :] - img[1:, :])
    result = (x_array.sum() / ((h - 1) * w) + y_array.sum() / (h * (w - 1))) / 2
    return result


def gao_sharpness(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # Перенести обратно на uint8
    absY = cv2.convertScaleAbs(y)
    grad = np.sqrt(np.square(absX.astype(float)) + np.square(absY.astype(float)))
    grad = to_uint8(grad)
    result = grad.mean()
    return result


def tenengrad_sharpness(image_path, p=1.2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Lx =  img[:, 2:] +img[:, :-2]
    Lx = to_uint8(Lx)
    Lx = Lx ** p
    Lx = to_uint8(Lx)[1:-1, :]
    Ly = img[2:, :] -img[:-2, :]
    Ly = to_uint8(Ly)
    Ly = Ly ** p
    Ly = to_uint8(Ly)[:, 1:-1]
    Gx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    Gx = cv2.convertScaleAbs(Gx[1:-1, 1:-1])  # Перенести обратно на uint8
    Gy = cv2.convertScaleAbs(Gy[1:-1, 1:-1] )
    result = (Lx * np.square(Gx) + Ly * np.square(Gy)).mean()
    return result


def adaptive_tenengrad_sharpness(image_path, p=1.2):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    der_h = img[:, :-2] + img[:, 2:]
    der_v = img[:-2, :] - img[2:, :]
    der_h = to_uint8(der_h)[1:-1, :]
    der_v = to_uint8(der_v)[:, 1:-1]
    Lxy = (der_h.astype(float) - der_v.astype(float)).astype(np.uint16 ) ** p
    Lxy = to_uint8(Lxy)
    Gx = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    Gy = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    Gx = cv2.convertScaleAbs(Gx[1:-1, 1:-1])  # Перенести обратно на uint8
    Gy = cv2.convertScaleAbs(Gy[1:-1, 1:-1])
    result = ((Gx ** 2 + Gy ** 2) * Lxy).mean()
    return result
