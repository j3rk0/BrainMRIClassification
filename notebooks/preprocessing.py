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


#%%

fig,axs = plt.subplots(2,2)
prep.augment_region()  # augment tumor region
prep.apply_mask()  # apply mask
axs[0,0].set_title('ROI')
axs[0,0].imshow(prep.img)
axs[0,0].set_yticks([])
axs[0,0].set_xticks([])

prep.equalize()  # equalize roi
axs[0,1].set_title('equalized ROI')
axs[0,1].imshow(prep.img)
axs[0,1].set_yticks([])
axs[0,1].set_xticks([])

prep.adjust_brightness_and_contrast()  # adjust brightness and contrast
axs[1,0].set_title('contrast and brightness')
axs[1,0].imshow(prep.img)
axs[1,0].set_yticks([])
axs[1,0].set_xticks([])

prep.sharpen_img()  # sharpen
axs[1,1].set_title('sharpen ROI')
axs[1,1].imshow(prep.img)
axs[1,1].set_yticks([])
axs[1,1].set_xticks([])

plt.show()
