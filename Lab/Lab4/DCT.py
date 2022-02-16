import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math


# 读取图像
def readImage(path):
    img = cv.imread(path, 0)
    img = img.astype('float')
    return img


# DCT变换
def DCT(img):
    f = cv.dct(img)
    return f


# DCT逆变换
def IDCT(f):
    iimg = cv.idct(f)
    iimg = np.abs(iimg)
    return iimg


def change(img, option='low'):
    size = img.shape[0]
    dst = np.zeros((size, size))
    dst[:128, :128] = img[:128, :128]
    if option == 'low':
        return dst
    elif option == 'high':
        return img - dst


def scale(img, scale=0.25):
    size = int(img.shape[0]*scale)
    dst = np.zeros((size, size))
    if size == 128:
        dst[:128, :128] = img[:128, :128] / 4
    else:
        dst[:128, :128] = img[:128, :128]
    return dst


# 展示结果
def showImage(img, res, iimg):
    plt.subplot(131), plt.imshow(img, 'gray'), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(132), plt.imshow(res, 'gray'), plt.title('DCT Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(iimg, 'gray'), plt.title('Inverse DCT Image')
    plt.axis('off')
    plt.show()


# 只进行DCT变换和DCT逆变换
def Normal(path):
    img = readImage(path)
    dct = DCT(img)
    res = np.log(np.abs(dct)+1)
    idct = IDCT(dct)
    showImage(img, res, idct)
    # cv.imwrite('result.png', idct, [cv.IMWRITE_PNG_COMPRESSION, 0])


# DCT变换后仅保留低频信号
def LowFrequency(path):
    img = readImage(path)
    dct = DCT(img)
    dct = change(dct, 'low')
    res = np.log(np.abs(dct)+1)
    idct = IDCT(dct)
    showImage(img, res, idct)


# DCT变换后仅保留高频信号
def HighFrequency(path):
    img = readImage(path)
    dct = DCT(img)
    dct = change(dct, 'high')
    res = np.log(np.abs(dct)+1)
    idct = IDCT(dct)
    showImage(img, res, idct)


# 缩放
def Scale(path, factor):
    img = readImage(path)
    dct = DCT(img)
    dct = scale(dct, factor)
    res = np.log(np.abs(dct)+1)
    idct = IDCT(dct)
    print(idct.shape)
    cv.imwrite('result.png', idct, [cv.IMWRITE_PNG_COMPRESSION, 0])
    showImage(img, res, idct)


def alpha(k, dimension):
    if k == 0:
        return math.sqrt(1/dimension)
    else:
        return math.sqrt(2/dimension)


def showMatrix(dimension):
    dct_matrix = np.zeros((dimension, dimension))
    for k in range(dimension):
        for l in range(dimension):
            dct_matrix[k, l] = alpha(k, dimension)*math.cos(math.pi*(l+0.5)*k/dimension)
    print(dct_matrix)
    plt.imshow(dct_matrix, 'gray')
    plt.title('DCT Matrix')
    plt.axis('off')
    plt.show()


file_path = 'Lena.png'

# 普通DCT变换
Normal(file_path)

# 保留低频
LowFrequency(file_path)

# 保留高频
HighFrequency(file_path)

# 缩放
Scale(file_path, 0.25)
Scale(file_path, 2)

# DCT矩阵
showMatrix(4)
showMatrix(8)
