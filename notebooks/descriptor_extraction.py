from time import time

import cv2
import numpy as np
import pandas as pd

from lib.descriptor import SiftKeypointsExtractor
from lib.preprocessing import Preprocessor

# %% Load metadata and initialize pipeline

dataset = pd.read_csv('data/features.csv',index_col=0)
keyp_extractor = SiftKeypointsExtractor(None)
preprocessor = Preprocessor(None, None)

# %%
descriptors = None
img_ids = []
img_labels = []
img_folds = []
est_time = None
t0 = 0

for i in range(dataset.shape[0]):
    t0 = time()
    print(f"loading {i + 1}/{dataset.shape[0]}")

    # LOAD DATA
    img_name = dataset.iloc[i, 0]
    label = dataset.iloc[i, 2]
    fold = dataset.iloc[i, 1]
    preprocessor.set_img(cv2.imread(f'data/images/mri/{img_name}.png', 0))
    preprocessor.set_mask(cv2.imread(f'data/images/mask/{img_name}.png', 0))

    # preprocess image
    preprocessor.preprocessing_pipeline()

    # extract keypoint
    keyp_extractor.set_img(preprocessor.img)
    keyp_extractor.extract_keyp()

    # get descriptors
    descs = keyp_extractor.get_keyp_desc()

    if descriptors is None:
        descriptors = descs
    else:
        descriptors = np.vstack((descriptors, descs))
    img_ids += [img_name] * descs.shape[0]
    img_labels += [label] * descs.shape[0]
    img_folds += [fold] * descs.shape[0]
#%%
# save data
df = pd.DataFrame(descriptors)

df.columns = [f'feat-{i}' for i in range(128)]
df['label'] = img_labels
df['names'] = img_ids
df['fold'] = img_folds
df.to_csv("data/descriptors_fb.csv")
