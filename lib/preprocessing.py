import numpy as np
from skimage.morphology import dilation, diamond
from skimage.transform import resize


def augment_region(mask):
    mask[mask == np.max(mask)] = 1
    mask[mask != 1] = 0
    footprint = diamond(radius=20)
    d_mask = dilation(mask, footprint)
    return d_mask


def apply_mask(img, mask):
    img[mask == 0] = 0

    temp = np.copy(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j] != 0:
                img[i, j] = temp[i, j]

    h = np.sum(mask, axis=0)
    w = np.sum(mask, axis=1)

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

    img[p3:p4, p1] = -1
    img[p3:p4, p2] = -1
    img[p3, p1:p2] = -1
    img[p4, p1:p2] = -1

    region = img[p3:p4, p1:p2]
    return resize(region, (512, 512), anti_aliasing=True)
