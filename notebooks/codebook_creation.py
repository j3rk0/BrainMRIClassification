import pickle as pkl
import datatable as dt
from sklearn.cluster import KMeans

# %% load data

df = dt.fread('data/descriptors.csv')
img_labels = df['label'].to_numpy().T[0]
fold = df['fold'].to_numpy().T[0]
img = df['names'].to_numpy().T[0]
descriptors = df[:, 1:-3]

# %% sift -> rootsift

descriptors /= (descriptors.to_numpy().sum(axis=1, keepdims=True) + 1e-7)

# %% split data
train_index = fold > 1

train_y = img_labels[train_index]
test_y = img_labels[~train_index]

train_X = descriptors[train_index, :]
test_X = descriptors[~train_index, :]

# %% apply clustering algorithm to build codebook

k = 50
km = KMeans(n_clusters=k)
print(f'clustering with k:{k}')
clust = km.fit_predict(train_X)
pkl.dump(km, open(f'models/kmeans_{k}.sk', 'wb'))
print('done')