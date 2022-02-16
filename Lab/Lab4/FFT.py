import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


# 读取图像
img = cv.imread('Lena.png', 0)


# 傅里叶变换
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)


# 计算幅值与相位
amplitude = np.log(np.abs(fshift)+1)
angle = np.angle(fshift)


# 傅里叶逆变换
ishift = np.fft.ifftshift(fshift)
iimg = np.fft.ifft2(ishift)
iimg = np.abs(iimg)


# 展示结果
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original Image')
plt.axis('off')
plt.subplot(122), plt.imshow(iimg, 'gray'), plt.title('Inverse Fourier Image')
plt.axis('off')
plt.show()

plt.subplot(121), plt.imshow(amplitude, 'gray'), plt.title('Fourier Image-Amplitude')
plt.axis('off')
plt.subplot(122), plt.imshow(angle, 'gray'), plt.title('Fourier Image-Angle')
plt.axis('off')
plt.show()
