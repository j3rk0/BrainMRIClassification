import pickle as pkl
from random import sample, seed

import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

seed(666)


def descriptors_to_bow(model, desc, index, labels):
    clust = model.predict(desc)
    i, j = 1, 0
    X = []
    y = []
    idx = []
    while i < index.shape[0]:  # take the sum of cluster responsability of each image
        while i < index.shape[0] and index[i] == index[j]:
            i += 1

        to_append = np.zeros(model.n_clusters)
        for h in range(j, i):
            to_append[clust[h]] += 1
        X.append(to_append)
        y.append(labels[j])
        idx.append(index[j])
        j = i
        i += 1
    return np.array(X), np.array(y), np.array(idx)


# %% LOAD DATA

df = dt.fread('data/descriptors.csv')
shape_df = dt.fread('data/shape_features.csv')
features = dt.fread('data/features.csv')[:, 1:]

img_labels = df['label'].to_numpy().T[0]
img_fold = df['fold'].to_numpy().T[0]
img_index = df['names'].to_numpy().T[0]
descriptors = df[:, 1:-3].to_numpy(type=np.float64)

descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)

# %% SPLIT TRAIN/TEST


mask = img_fold < 5

train_labels = img_labels[mask]
train_desc = descriptors[mask]
train_idx = img_index[mask]

test_labels = img_labels[~mask]
test_desc = descriptors[~mask]
test_idx = img_index[~mask]

del mask, descriptors, img_labels, img_index, img_fold, df

# %% convert DESCRIPTORS to BOW

k_means = pkl.load(open('models/kmeans_50.sk', 'rb'))
k = k_means.n_clusters

X_train, y_train, idx_train = descriptors_to_bow(k_means, train_desc, train_idx, train_labels)
X_test, y_test, idx_test = descriptors_to_bow(k_means, test_desc, test_idx, test_labels)

# %% merge BOW, SHAPE and SLICE features [don't run if you want to test bow only]

train_shapes = shape_df[[i in idx_train for i in shape_df['img'].to_numpy()], :]
test_shapes = shape_df[[i in idx_test for i in shape_df['img'].to_numpy()], :]

X_train = np.concatenate((X_train, train_shapes[:, 2:].to_numpy()), axis=1)
X_test = np.concatenate((X_test, test_shapes[:, 2:].to_numpy()), axis=1)

del train_desc, train_idx, train_shapes, test_desc, test_idx, test_shapes, idx_train, idx_test, shape_df, k_means
del train_labels, test_labels, features

# %% SUBSAMPLE DATA TO BALANCE CLASSES

l1_map = y_train == 1
l2_map = y_train == 2
l3_map = y_train == 3

l1_X = X_train[l1_map, :]
l2_X = X_train[l2_map, :]
l3_X = X_train[l3_map, :]

l1_y = y_train[l1_map]
l2_y = y_train[l2_map]
l3_y = y_train[l3_map]

selected_l1 = sample(range(l1_y.shape[0]), 500)
selected_l2 = sample(range(l2_y.shape[0]), 500)
selected_l3 = sample(range(l3_y.shape[0]), 500)

X_train = dt.rbind(dt.Frame(l1_X[selected_l1, :]),
                   dt.Frame(l2_X[selected_l2, :]),
                   dt.Frame(l3_X[selected_l3, :])).to_numpy()

y_train = np.array([1] * 500 + [2] * 500 + [3] * 500)

del l1_X, l1_y, l1_map, selected_l1, l2_map, l2_y, l2_X, selected_l2, l3_X, l3_y, l3_map, selected_l3, sample

# %% PREPROCESS DATA

tf_idf = TfidfTransformer()
tf_idf.fit(X_train[:, :k])

X_train = np.concatenate((tf_idf.transform(X_train[:, :k]).toarray(), X_train[:, k:]), axis=1)
X_test = np.concatenate((tf_idf.transform(X_test[:, :k]).toarray(), X_test[:, k:]), axis=1)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train -= 1
y_test -= 1

pkl.dump(tf_idf, open(f'models/tf_idf_{k}.sk', 'wb'))
pkl.dump(scaler, open(f'models/scaler_{k}.sk', 'wb'))

del tf_idf, scaler

# %% VISUALIZE DATA
x_toplot = PCA(n_components=2).fit_transform(
    np.concatenate((X_train, X_test), axis=0))
plt.scatter(x_toplot[:, 0], x_toplot[:, 1], c=np.concatenate((y_train, y_test)))
plt.show()

del x_toplot

# %% TRAIN AND VALIDATE CLASSIFIER

xgb = XGBClassifier(num_parallel_tree=50,
                    learning_rate=1,
                    max_depth=5,
                    booster='gbtree',
                    subsample=.8,
                    colsample_bynode=.8)
xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)
print(f"accuracy:{accuracy_score(y_test, pred)}")
print(f"confusion matrix:\n{confusion_matrix(y_test, pred)}")
