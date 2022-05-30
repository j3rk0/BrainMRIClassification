import numpy as np
from skimage.morphology import dilation, diamond
from skimage.transform import resize
from skimage.exposure import equalize_adapthist
from skimage.filters import unsharp_mask
import cv2


class Preprocessor:

    def __init__(self, img, mask):
        self.img = img
        self.mask = mask

    def set_img(self, img):
        self.img = img

    def set_mask(self, mask):
        self.mask = mask

    def get_prerprocessed_img(self):
        self.mask = self.augment_region()
        self.img = self.apply_mask()
        self.img = np.uint8(self.equalize())
        self.img = self.adjust_brightness_and_contrast()
        return self.sharpen_img()

    def sharpen_img(self):
        sharpened = unsharp_mask(self.img, 7, 1)
        return np.uint8(cv2.normalize(sharpened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

    def augment_region(self):
        self.mask[self.mask == np.max(self.mask)] = 1
        self.mask[self.mask != 1] = 0
        footprint = diamond(radius=10)
        d_mask = dilation(self.mask, footprint)
        return d_mask

    def apply_mask(self):
        self.mask[self.mask == np.max(self.mask)] = 1
        self.mask[self.mask != 1] = 0

        self.img[self.mask == 0] = 0

        temp = np.copy(self.img)

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if self.mask[i, j] != 0:
                    self.img[i, j] = temp[i, j]

        h = np.sum(self.mask, axis=0)
        w = np.sum(self.mask, axis=1)

        p1 = None
        p2 = None
        p3 = None
        p4 = None

        for i in range(h.shape[0]):
            if i < 5:
                continue
            if h[i] != 0 and p1 is None:
                p1 = i - 1
            if w[i] != 0 and p3 is None:
                p3 = i - 1
            if h[i] == 0 and p1 is not None and p2 is None:
                p2 = i + 1
            if w[i] == 0 and p3 is not None and p4 is None:
                p4 = i + 1

        self.img[p3:p4, p1] = -1
        self.img[p3:p4, p2] = -1
        self.img[p3, p1:p2] = -1
        self.img[p4, p1:p2] = -1

        region = self.img[p3:p4, p1:p2]
        return resize(region, (512, 512), anti_aliasing=True)

    def adjust_brightness_and_contrast(self, clip_hist_percent=1):
        # Calculate grayscale histogram
        hist = cv2.calcHist([self.img], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = [float(hist[0])]
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        return cv2.convertScaleAbs(self.img, alpha=alpha, beta=beta)

    def equalize(self):
        equalized_img = equalize_adapthist(self.img)
        return cv2.normalize(equalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
