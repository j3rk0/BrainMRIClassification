import cv2
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist

from lib.preprocessing import augment_region, apply_mask

img = cv2.imread('data/images/mri/1.png', 0)
mask = cv2.imread('data/images/mask/1.png', 0)
plt.gray()
# %%

mask_augmented = augment_region(mask)

# %%

masked_img = apply_mask(img, mask_augmented)
plt.imshow(masked_img)
plt.show()

# %%

equalized_img = equalize_adapthist(masked_img)
equalized_img = cv2.normalize(equalized_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
plt.imshow(equalized_img)
plt.show()

#%%
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi


