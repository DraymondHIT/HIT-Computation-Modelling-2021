import numpy as np
import cv2
from matplotlib import pyplot as plt


# O(r)
def _quickMedianFiltering(img, r):
    # 初始化参数
    img = np.pad(img, (r, r))
    rows = img.shape[0]
    cols = img.shape[1]
    L = 2 * r + 1
    num = L * L
    cidx = num / 2 + 0.5
    out = np.zeros([rows, cols])

    for i in range(r, rows - r):
        # 计算第一列
        H = [0] * 256
        # 生成直方图
        tmp = img[i - r:i + r + 1, 0:L].flatten()
        for n in range(num):
            H[tmp[n]] += 1

        # 累积直方图
        n = 0
        for med in range(256):
            n += H[med]
            if n >= cidx:
                out[i, r] = med
                break

        # 后续计算
        for j in range(1 + r, cols - r):
            for m in range(-r, r + 1):
                H[img[i + m, j - 1 - r]] -= 1
                if img[i + m, j - 1 - r] <= med:
                    n -= 1
                H[img[i + m, j + r]] += 1
                if img[i + m, j + r] <= med:
                    n += 1

            if n < cidx:
                while n < cidx:
                    med += 1
                    n += H[med]
            elif n > cidx:
                while n - H[med] >= cidx:
                    n -= H[med]
                    med -= 1
            out[i, j] = med

    return out[r:rows-r, r:cols-r]


def getMedianValue(H, cidx):
    n = 0
    for med in range(256):
        n += H[med]
        if n >= cidx:
            break
    return med

# O(1) 但依赖于位深，实际速度不如上面的算法
def quickMedianFiltering(img, r):
    # 初始化参数
    img = np.pad(img, (r, r))
    rows = img.shape[0]
    cols = img.shape[1]
    L = 2 * r + 1
    num = L * L
    cidx = num / 2 + 0.5
    out = np.zeros([rows, cols])

    # 初始化直方图
    H = np.zeros((cols, 256))
    for i in range(2*r+1):
        for j in range(len(H)):
            H[j][img[i, j]] += 1

    for i in range(r+1, rows - r):
        total = np.zeros((1, 256))
        for j in range(2*r+1):
            H[j][img[i-r-1, j]] -= 1
            H[j][img[i+r, j]] += 1
            total += H[j].reshape(1, -1)
        out[i, r] = getMedianValue(total.squeeze(), cidx)

        # 后续计算
        for j in range(r+1, cols-r):
            H[j+r][img[i-r-1, j+r]] -= 1
            H[j+r][img[i+r, j+r]] += 1
            total += H[j+r].reshape(1, -1)
            total -= H[j-r-1].reshape(1, -1)

            out[i, j] = getMedianValue(total.squeeze(), cidx)

    return out[r:rows-r, r:cols-r]


img = cv2.imread('Lena.tiff', 0)
_img = quickMedianFiltering(img, 1)
plt.imshow(_img, 'gray')
plt.axis('off')
plt.show()
