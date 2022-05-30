import numpy as np
from lib.filter_bank import get_filter_bank
from skimage.exposure import histogram
from skimage.feature import corner_shi_tomasi, corner_peaks
import cv2
from skimage.feature import SIFT


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
        extractor = SIFT()
        extractor.detect_and_extract(self.img)
        self.keyp_desc = extractor.descriptors
        self.keyp_loc = extractor.keypoints


class SurfKeypointsExtractor(KeyPointExtractor):

    def extract_keyp(self):
        pass
        # surf = xfeatures2d.SURF_create()
        # self.keyp_desc, self.keyp_loc = surf.detectAndCompute(self.img, None)


class FilterBankExtractor(KeyPointExtractor):
    def extract_keyp(self):
        self.img = cv2.normalize(self.img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        self.keyp_loc = corner_peaks(corner_shi_tomasi(self.img, sigma=4), min_distance=8)

        self.keyp_desc = []
        for i in range(self.keyp_loc.shape[0]):
            self.keyp_desc.append(self._kp_to_descriptor(i))

    def _kp_to_descriptor(self, keyp_index):
        kp = self.keyp_loc[keyp_index, :]
        patch = self.img[kp[0] - 4:kp[0] + 5, kp[1] - 4:kp[1] + 5]
        hist = cv2.calcHist(patch, [0], None, [16], [0, 256]).reshape(16)
        hist /= np.sum(hist)
        resp = []
        for ker in get_filter_bank():
            resp.append(np.sum(ker * patch))
        return np.concatenate((hist, np.array(resp)), axis=0)


def group_hist(hist, nbins=16):
    ret = np.zeros(nbins)
    for i in range(nbins):
        for j in range(hist.shape[0] // nbins):
            ret[i] += hist[nbins * i + j]
    return ret / np.sum(ret)
