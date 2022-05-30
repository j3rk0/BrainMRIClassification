from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from umap import UMAP
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from lib.descriptor import FilterBankExtractor, SiftKeypointsExtractor
from sklearn.metrics import homogeneity_score
from lib.filter_bank import get_filter_bank
from lib.preprocessing import Preprocessor

# %%
img = cv2.imread('data/images/mri/1.png', 0)
mask = cv2.imread('data/images/mask/1.png', 0)

preprocessor = Preprocessor(img, mask)

img = preprocessor.get_prerprocessed_img()

plt.gray()
plt.imshow(img)
plt.show()

# %%

dataset = pd.read_csv('data/features.csv')
keyp_extractor = SiftKeypointsExtractor(None)
preprocessor = Preprocessor(None, None)
descriptors = None
labels = []

for i in range(300):
    print(f"loaded {i}/300")
    index = randint(0, dataset.shape[0] - 1)
    img_name = dataset.iloc[index, 0]
    label = dataset.iloc[index, 2]

    preprocessor.set_img(cv2.imread(f'data/images/mri/{img_name}.png', 0))
    preprocessor.set_mask(cv2.imread(f'data/images/mask/{img_name}.png', 0))

    img = preprocessor.get_prerprocessed_img()

    keyp_extractor.set_img(img)
    keyp_extractor.extract_keyp()

    descs = keyp_extractor.get_keyp_desc()

    if descriptors is None:
        descriptors = descs
    else:
        descriptors = np.vstack((descriptors, descs))
    labels += [label] * descs.shape[0]

# %%

embedded_desc = PCA(n_components=2).fit_transform(np.array(descriptors))

plt.scatter(x=embedded_desc[:, 0], y=embedded_desc[:, 1], c=labels,cmap='set1')
plt.show()
# %%
clust = KMeans(n_clusters=300).fit_predict(np.array(descriptors))

plt.scatter(x=embedded_desc[:, 0], y=embedded_desc[:, 1], c=clust, cmap='Pastel1')
plt.show()
# %%
from sklearn.metrics import homogeneity_score

print(homogeneity_score(labels, clust))
