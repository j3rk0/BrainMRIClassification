import cv2
import numpy as np
import sys
from scipy.ndimage import convolve

filt_bank = {
    'log': {'ksiz': 7, 'std': 5},
    'gauss': {'ksiz': 7, 'sig': 5}
}


def getGaussianKernel(ker=5, sig=1.):
    ax = np.linspace(-(ker - 1) / 2., (ker - 1) / 2., ker)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def getLogKernel(ker, std):
    siz = (1 - ker) // 2
    x = y = np.linspace(siz, -siz, ker)
    x, y = np.meshgrid(x, y)
    arg = -(x ** 2 + y ** 2) / (2 * std ** 2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h / h.sum() if h.sum() != 0 else h
    h1 = h * (x ** 2 + y ** 2 - 2 * std ** 2) / (std ** 4)
    return h1 - h1.mean()


def getGaborKernels():
    deg = 0
    kernels = []
    for i in range(4):
        deg += 45
        theta = np.radians(deg)
        for sig in [1, 2]:
            for lamb in [3, 10]:
                kernels.append(cv2.getGaborKernel((9, 9), sig, theta, lamb, 0.25, ktype=cv2.CV_64F))
    return kernels


def get_filter_bank():
    log = getLogKernel(9, 2)
    gauss = getGaussianKernel(9, 2)
    gabor = getGaborKernels()
    return gabor + [log, gauss]
