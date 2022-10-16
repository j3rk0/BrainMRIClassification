import cv2
import matplotlib.pyplot as plt
from lib.preprocessing import Preprocessor
plt.gray()

# %% EXAMPLE PREPROCESSING PIPELINE

# load image
img = cv2.imread('data/images/mri/1.png', 0)
mask = cv2.imread('data/images/mask/1.png', 0)
prep = Preprocessor(img, mask)
plt.title('original image')
plt.imshow(img)
plt.show()

prep.augment_region()  # augment tumor region
prep.apply_mask()  # apply mask
plt.title('ROI')
plt.imshow(prep.img)
plt.show()

prep.equalize()  # equalize roi
plt.title('equalized ROI')
plt.imshow(prep.img)
plt.show()

prep.adjust_brightness_and_contrast()  # adjust brightness and contrast
plt.title('contrast and brightness adjusted')
plt.imshow(prep.img)
plt.show()

prep.sharpen_img()  # sharpen
plt.title('sharpen roi')
plt.imshow(prep.img)
plt.show()
