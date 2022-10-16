import numpy as np
import pandas as pd
from cv2 import imread
from skimage.measure import regionprops

# %% load metadata and initialize dataframe
res = {'img': [],
       'area': [],
       'perimeter': [],
       'centroid_x': [],
       'centroid_y': [],
       'slice_area': [],
       'slice_center_x': [],
       'slice_center_y': []}

features = pd.read_csv('data/features.csv', index_col=0)

# %%

for i in range(features.shape[0]):
    print(f'analyzing image {i + 1}/{features.shape[0]}')
    # load data
    img_name = features.iloc[i, 0]
    mask = imread(f'data/images/mask/{img_name}.png', 0)
    # preprocess mask
    mask[mask == 30] = 0
    mask[mask > 0] = 1
    # calculate props
    props = regionprops(mask)[0]
    centroid = np.array(props.centroid)
    res['img'].append(img_name)

    res['area'].append(props.area / (512 ** 2))
    res['perimeter'].append(props.perimeter)
    res['centroid_x'].append(centroid[0] / 512)
    res['centroid_y'].append(centroid[1] / 512)

    # extract the features of the slice
    img = imread(f"data/images/mri/{img_name}.png", 0)  # read image
    img = (img - img.min()) / (img.max() - img.min())  # normalize
    img[img < .2] = 0  # apply treshold
    img[img > 0] = 1
    slice_center = np.array(regionprops(img.astype(np.uint8))[0].centroid)

    res['slice_area'].append(img.sum() / (512 ** 2))
    res['slice_center_x'].append(slice_center[0] / 512)
    res['slice_center_y'].append(slice_center[1] / 512)


# %% save to csv
pd.DataFrame.from_dict(res).to_csv('data/shape_features.csv')
print('done')
