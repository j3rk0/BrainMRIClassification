import cv2
import numpy as np
import sys
from scipy.ndimage import convolve


def getLogKernel(siz, std):
    x = y = np.linspace(-siz, siz, 2 * siz + 1)
    x, y = np.meshgrid(x, y)
    arg = -(x ** 2 + y ** 2) / (2 * std ** 2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h / h.sum() if h.sum() != 0 else h
    h1 = h * (x ** 2 + y ** 2 - 2 * std ** 2) / (std ** 4)
    return h1 - h1.mean()


def get_filter_bank():
    gauss = cv2.getGaussianKernel()
    gabor = cv2.getGaborKernel()
    log = getLogKernel()
