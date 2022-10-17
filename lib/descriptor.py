import cv2
import numpy as np
from skimage.feature import ORB
from skimage.feature import SIFT
from skimage.feature import corner_shi_tomasi, corner_peaks


class KeyPointExtractor:

    def __init__(self, img):
        self.img = img
        self.keyp_desc = None
        self.keyp_loc = None
        pass

    def set_img(self, img):
        self.img = img

    def get_keyp_desc(self):
        return self.keyp_desc

    def get_keyp_loc(self):
        return self.keyp_loc

    def extract_keyp(self):
        pass


class SiftKeypointsExtractor(KeyPointExtractor):
    def extract_keyp(self):
        extractor = SIFT(upsampling=1, n_scales=6, sigma_min=3)
        extractor.detect_and_extract(self.img)
        self.keyp_desc = extractor.descriptors
        self.keyp_loc = extractor.keypoints


class ORBKeypoitsExtractor(KeyPointExtractor):
    def extract_keyp(self):
        extractor = ORB()
        extractor.detect_and_extract(self.img)
        self.keyp_desc = extractor.descriptors
        self.keyp_loc = extractor.keypoints


def group_hist(hist, nbins=16):
    ret = np.zeros(nbins)
    for i in range(nbins):
        for j in range(hist.shape[0] // nbins):
            ret[i] += hist[nbins * i + j]
    return ret / np.sum(ret)
