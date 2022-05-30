import cv2
import matplotlib.pyplot as plt
import numpy as np
from lib.preprocessing import equalize, augment_region, apply_mask, adjust_brightness_and_contrast
from skimage.filters import unsharp_mask
# %%
img = cv2.imread('data/images/mri/1.png', 0)
mask = cv2.imread('data/images/mask/1.png', 0)
plt.gray()
# %%

mask_augmented = augment_region(mask)

# %%

masked_img = apply_mask(img, mask_augmented)
plt.imshow(masked_img)
plt.show()

# %% equalize

equalized_img = equalize(masked_img)
equalized_img = np.uint8(cv2.normalize(equalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
plt.imshow(equalized_img)
plt.show()
#%% adjust brightness and contrast

adjusted_img = adjust_brightness_and_contrast(equalized_img)
plt.imshow(adjusted_img)
plt.show()

#%% sharpen

sharpen_img = unsharp_mask(adjusted_img,7,1)
sharpen_img = np.uint8(cv2.normalize(sharpen_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))
plt.imshow(sharpen_img)
plt.show()


# %% log and gaussian filter

from lib.filter_bank import get_filter_bank

kernels = get_filter_bank()

fig, axes = plt.subplots(5, 8)

h = 0
for i in range(4):
    for j in range(8):
        axes[i, j].imshow(kernels[h])
        h += 1

axes[4, 0].imshow(kernels[-2])
axes[4, 1].imshow(kernels[-1])

plt.show()

# %%





# %%
