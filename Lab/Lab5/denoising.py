import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import random
import csv


def add_salt_pepper_noise(image, psn, p, low, high):
    assert psn < 100 and 0 < p < 1
    height, width = image.shape[:2]
    dst = np.array(image, copy=True)
    salt_pepper_num = int(height * width * psn / 100.)
    salt_pepper_position = set()
    while salt_pepper_num > 0:
        x = random.randint(0, height - 1)
        y = random.randint(0, width - 1)
        if not (x, y) in salt_pepper_position:
            salt_pepper_position.add((x, y))
            salt_pepper_num -= 1
    for x, y in salt_pepper_position:
        num = np.random.uniform(0, 1)
        if num < p:
            dst[x, y] = low
        else:
            dst[x, y] = high
    return dst


def add_Gaussian_noise(image, var=0.005):
    # 归一化
    image = np.array(image/255, dtype=float)
    # 产生高斯噪声
    noise = np.random.normal(0, var ** 0.5, image.shape)
    dst = image + noise
    dst = np.uint8(dst*255)
    return dst


def show_image_salt(_img, _noise, _dst, psn, psnr, ssim, ksize):
    plt.subplot(131)
    plt.imshow(_img, 'gray')
    plt.title('original')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(_noise, 'gray')
    plt.title(f'psn = {psn}')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(_dst, 'gray')
    plt.title(f'psnr = {psnr:.4f}\nssim = {ssim:.4f}\nksize = {ksize}')
    plt.axis('off')
    plt.show()


def show_image_Gaussian(_img, _noise, _dst, var, psnr, ssim, ksize, sigma):
    plt.subplot(131)
    plt.imshow(_img, 'gray')
    plt.title('original')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(_noise, 'gray')
    plt.title(f'var = {var}')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(_dst, 'gray')
    plt.title(f'psnr = {psnr:.4f}\nssim = {ssim:.4f}\nksize = {ksize}\nsigma = {sigma}')
    plt.axis('off')
    plt.show()


def show_image(img1, img2, img3):
    plt.subplot(131)
    plt.imshow(img1, 'gray')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(img2, 'gray')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(img3, 'gray')
    plt.axis('off')
    plt.show()


def salt_pepper_noise_filter(image, step, p=0.5, low=0, high=255):
    # 添加椒盐噪声
    _img_noise = []
    for psn in range(5, 96, step):
        _img = add_salt_pepper_noise(img, psn=psn, p=p, low=low, high=high)
        _img_noise.append(_img)

    # 中值滤波
    _img_blur = []
    for _img in _img_noise:
        _img_medium = [cv2.medianBlur(_img, i) for i in range(3, 62, 2)]
        _img_blur.append(_img_medium)

    best_ksize = []

    with open('result.csv', 'w') as csv_file:
        fieldnames = ['psn', 'psnr', 'ssim', 'ksize']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(_img_noise)):
            _psn = 5 + step * i
            _psnr_max = 0
            _ssim_max = 0
            _ksize_max = 0
            for j in range(30):
                _psnr = psnr(img, _img_blur[i][j])
                _ssim = ssim(img, _img_blur[i][j])
                _ksize = 3 + 2 * j
                if _psnr > _psnr_max and _ssim > _ssim_max:
                    _psnr_max = _psnr
                    _ssim_max = _ssim
                    _ksize_max = _ksize
                writer.writerow({'psn': 5 + step * i, 'psnr': _psnr, 'ssim': _ssim, 'ksize': _ksize})
            best_ksize.append(_ksize_max)

    # 出图
    for i in range(len(_img_noise)):
        _psn = 5 + step * i
        _psnr = psnr(img, _img_blur[i][(best_ksize[i]-3)//2])
        _ssim = ssim(img, _img_blur[i][(best_ksize[i]-3)//2])
        show_image_salt(img, _img_noise[i], _img_blur[i][(best_ksize[i]-3)//2], psn=_psn, psnr=_psnr, ssim=_ssim, ksize=best_ksize[i])


def Gaussian_noise_filter(image, var, sigma):
    # 添加高斯噪声
    _img_noise = add_Gaussian_noise(image, var=var)

    # 高斯滤波
    _psnr_max = 0
    _ssim_max = 0
    _ksize_max = 0
    for i in range(3, 20, 2):
        _dst = cv2.GaussianBlur(_img_noise, (i, i), sigma)
        _psnr = psnr(img, _dst)
        _ssim = ssim(img, _dst)
        if _psnr > _psnr_max and _ssim > _ssim_max:
            _psnr_max = _psnr
            _ssim_max = _ssim
            _ksize_max = i
    _dst = cv2.GaussianBlur(_img_noise, (_ksize_max, _ksize_max), sigma)
    show_image_Gaussian(image, _img_noise, _dst, var, _psnr_max, _ssim_max, _ksize_max, sigma)

    # 中值滤波
    _psnr_max = 0
    _ssim_max = 0
    _ksize_max = 0
    for i in range(3, 20, 2):
        _dst = cv2.medianBlur(_img_noise, i)
        _psnr = psnr(img, _dst)
        _ssim = ssim(img, _dst)
        if _psnr > _psnr_max and _ssim > _ssim_max:
            _psnr_max = _psnr
            _ssim_max = _ssim
            _ksize_max = i
    _dst = cv2.medianBlur(_img_noise, _ksize_max)
    show_image_Gaussian(image, _img_noise, _dst, var, _psnr_max, _ssim_max, _ksize_max, None)


def mixed_noise_filter(image, psn, p=0.5, low=0, high=255, var=0.005, sigma=1.5):
    # 添加混合噪声
    _img_noise = add_Gaussian_noise(image, var=var)
    _img_noise = add_salt_pepper_noise(image, psn, p, low, high)

    # Gaussian only
    _dst_Gaussian = cv2.GaussianBlur(_img_noise, (3, 3), sigma)
    _psnr = psnr(image, _dst_Gaussian)
    print("Gaussian only: ", _psnr)

    # median only
    _dst_median = cv2.medianBlur(_img_noise, 3)
    _psnr = psnr(image, _dst_median)
    print("median only: ", _psnr)

    # mixed
    _dst = cv2.medianBlur(_img_noise, 3)
    _dst = cv2.GaussianBlur(_dst, (3, 3), sigma)
    _psnr = psnr(image, _dst)
    print("mixed: ", _psnr)

    show_image(_dst_Gaussian, _dst_median, _dst)


def OTSU(img_gray, GrayScale):
    img_gray = np.array(img_gray).ravel().astype(np.uint8)
    th = 0.0
    # 总的像素数目
    PixSum = img_gray.size
    # 各个灰度值的像素数目
    PixCount = np.zeros(GrayScale)
    # 各灰度值所占总像素数的比例
    PixRate = np.zeros(GrayScale)
    # 统计各个灰度值的像素个数
    for i in range(PixSum):
        # 默认灰度图像的像素值范围为GrayScale
        Pixvalue = img_gray[i]
        PixCount[Pixvalue] = PixCount[Pixvalue] + 1

    # 确定各个灰度值对应的像素点的个数在所有的像素点中的比例
    for j in range(GrayScale):
        PixRate[j] = PixCount[j] * 1.0 / PixSum
    Max_var = 0
    # 确定最大类间方差对应的阈值
    for i in range(1, GrayScale):
        u1_tem = 0.0
        u2_tem = 0.0
        # 背景像素的比列
        w1 = np.sum(PixRate[:i])
        # 前景像素的比例
        w2 = 1.0 - w1
        if w1 == 0 or w2 == 0:
            pass
        else:
            # 背景像素的平均灰度值
            for m in range(i):
                u1_tem = u1_tem + PixRate[m] * m
            u1 = u1_tem * 1.0 / w1
            # 前景像素的平均灰度值
            for n in range(i, GrayScale):
                u2_tem = u2_tem + PixRate[n] * n
            u2 = u2_tem / w2
            # 类间方差公式
            tem_var = w1 * w2 * np.power((u1 - u2), 2)
            if Max_var < tem_var:
                Max_var = tem_var
                th = i
    return th


def threshold(image, thresh):
    height, width = image.shape
    image_prime = np.array(image, copy=True)
    for i in range(height):
        for j in range(width):
            if image_prime[i, j] <= thresh:
                image_prime[i, j] = 0
            else:
                image_prime[i, j] = 255
    return image_prime


def halftone(image, padding):
    height, width = image.shape
    image_prime = np.array(image, copy=True)
    image_prime = np.pad(image_prime, (padding, padding))/255.
    dst = np.zeros((image_prime.shape[0], image_prime.shape[1]))
    K = np.array([
        [0, 0, 0, 7, 5],
        [3, 5, 7, 5, 3],
        [1, 3, 5, 3, 1]
    ])/48.
    for i in range(padding, padding+height):
        for j in range(padding, padding+width):
            if image_prime[i, j] < 0.5:
                dst[i, j] = 0
            else:
                dst[i, j] = 1

            error = image_prime[i, j] - dst[i, j]
            image_prime[i:i+3, j-2:j+3] += error*K
    return np.uint8(image_prime[padding:padding+height, padding:padding+width]*255)


def LowFrequency(image):
    image = cv2.dct(image.astype(np.float64))
    dst = np.zeros((image.shape[0], image.shape[1]))
    dst[:128, :128] = image[:128, :128]
    return np.abs(cv2.idct(dst)).astype(np.uint8)


# 步长
step = 10

# 图像读取
img = cv2.imread('Lena.tiff', 0)

# 椒盐噪声滤除
salt_pepper_noise_filter(img, step)

# 高斯噪声滤除
Gaussian_noise_filter(img, var=0.005, sigma=1.)

# 混合噪声滤除
mixed_noise_filter(img, psn=10, var=0.005, sigma=1.5)

# 大津阈值
th = OTSU(img, 256)
print(th)

# 二值化
_img_thresh = threshold(img, th)
plt.imshow(_img_thresh, 'gray')
plt.title(f'psnr = {psnr(img, _img_thresh)}')
plt.axis('off')
plt.show()


# 半色调
_img_halftone = halftone(img, 2)
plt.imshow(_img_halftone, 'gray')
plt.title(f'psnr = {psnr(img, _img_halftone)}')
plt.axis('off')
plt.show()


# 逆半色调
_img_inverse = LowFrequency(_img_halftone)
print(psnr(img, _img_inverse))
plt.imshow(_img_inverse, 'gray')
plt.title(f'psnr = {psnr(img, _img_inverse)}')
plt.axis('off')
plt.show()
