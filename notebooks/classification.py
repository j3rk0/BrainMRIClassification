import pickle as pkl
from random import sample, seed
from mlxtend.plotting import plot_confusion_matrix
import datatable as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

df = dt.fread('data/kp_descriptors.csv')
shape_df = dt.fread('data/shape_features.csv')
features = dt.fread('data/features.csv')[:, 1:]

img_labels = df['label'].to_numpy().T[0]
img_fold = df['fold'].to_numpy().T[0]
img_index = df['names'].to_numpy().T[0]
descriptors = df[:, 1:-3].to_numpy(type=np.float64)

descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
descriptors = np.sqrt(descriptors)

# %% SPLIT TRAIN/TEST


mask = img_fold != 1

train_labels = img_labels[mask]
train_desc = descriptors[mask]
train_idx = img_index[mask]

test_labels = img_labels[~mask]
test_desc = descriptors[~mask]
test_idx = img_index[~mask]

del mask, descriptors, img_labels, img_index, df, img_fold

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

del train_desc, train_idx, train_shapes, test_desc, test_idx, test_shapes
del train_labels, test_labels

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

X_train_balanced = dt.rbind(dt.Frame(l1_X[selected_l1, :]),
                            dt.Frame(l2_X[selected_l2, :]),
                            dt.Frame(l3_X[selected_l3, :])).to_numpy()

y_train_balanced = np.array([1] * 500 + [2] * 500 + [3] * 500) - 1

del l1_X, l1_y, l1_map, selected_l1, l2_map, l2_y, l2_X, selected_l2, l3_X, l3_y, l3_map, selected_l3, sample

# %% PREPROCESS DATA

tf_idf = TfidfTransformer()
tf_idf.fit(X_train_balanced[:, :k])
# %%
X_train_balanced = np.concatenate((tf_idf.transform(X_train_balanced[:, :k]).toarray(), X_train_balanced[:, k:]),
                                  axis=1)
X_train = np.concatenate((tf_idf.transform(X_train[:, :k]).toarray(), X_train[:, k:]), axis=1)
X_test = np.concatenate((tf_idf.transform(X_test[:, :k]).toarray(), X_test[:, k:]), axis=1)

scaler = MinMaxScaler()
X_train_balanced = scaler.fit_transform(X_train_balanced)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train -= 1
y_test -= 1

# %%
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
xgb.fit(X_train_balanced, y_train_balanced)

pred = xgb.predict(X_test)
print(f"accuracy:{accuracy_score(y_test, pred)}")
print(f"confusion matrix:\n{confusion_matrix(y_test, pred)}")

# %%
plot_confusion_matrix(confusion_matrix(y_test, pred))
plt.show()

# %% PLOTTING

dataset = np.concatenate((X_train, X_test), axis=0)
target = np.concatenate((y_train, y_test))
p1 = dataset[target < 2, :]
p2 = dataset[target > 0, :]
p3 = dataset[target != 1, :]

p1 = PCA(n_components=2).fit_transform(p1)
p2 = PCA(n_components=2).fit_transform(p2)
p3 = PCA(n_components=2).fit_transform(p3)

fig, axes = plt.subplots(1, 3)
fig.set_dpi(300)
fig.suptitle('first two PCA components of image descriptors vectorial space')
fig.set_size_inches(18.5, 10.5, forward=True)
axes[0].scatter(x=p1[:, 0], y=p1[:, 1], c=target[target < 2])
axes[0].set_title('Meningioma vs Glioma')
axes[0].set(xlabel='pca-1', ylabel='pca-2')
axes[1].scatter(x=p2[:, 0], y=p2[:, 1], c=target[target > 0])
axes[1].set_title('Glioma vs Pituritary')
axes[1].set(xlabel='pca-1', ylabel='pca-2')
axes[2].scatter(x=p3[:, 0], y=p3[:, 1], c=target[target != 1])
axes[2].set_title('Meningioma vs Pituritary')
axes[2].set(xlabel='pca-1', ylabel='pca-2')
plt.show()
# %%

fig, axes = plt.subplots(1, 3)
fig.set_dpi(300)
fig.suptitle('first two PCA components of bow image descriptors')
fig.set_size_inches(18.5, 10.5, forward=True)
axes[0].scatter(x=p1[:, 0], y=p1[:, 1], c=target[target < 2])
axes[0].set_title('Meningioma vs Glioma')
axes[0].set(xlabel='pca-1', ylabel='pca-2')
axes[1].scatter(x=p2[:, 0], y=p2[:, 1], c=target[target > 0])
axes[1].set_title('Glioma vs Pituritary')
axes[1].set(xlabel='pca-1', ylabel='pca-2')
axes[2].scatter(x=p3[:, 0], y=p3[:, 1], c=target[target != 1])
axes[2].set_title('Meningioma vs Pituritary')
axes[2].set(xlabel='pca-1', ylabel='pca-2')
plt.show()
# %%

fig, axes = plt.subplots(1, 3)
fig.set_dpi(300)
fig.suptitle('first two PCA components of shape and slice features')
fig.set_size_inches(18.5, 10.5, forward=True)
axes[0].scatter(x=p1[:, 0], y=p1[:, 1], c=target[target < 2])
axes[0].set_title('Meningioma vs Glioma')
axes[0].set(xlabel='pca-1', ylabel='pca-2')
axes[1].scatter(x=p2[:, 0], y=p2[:, 1], c=target[target > 0])
axes[1].set_title('Glioma vs Pituritary')
axes[1].set(xlabel='pca-1', ylabel='pca-2')
axes[2].scatter(x=p3[:, 0], y=p3[:, 1], c=target[target != 1])
axes[2].set_title('Meningioma vs Pituritary')
axes[2].set(xlabel='pca-1', ylabel='pca-2')
plt.show()

# %%
plt.pie([xgb.feature_importances_[:-7].sum() * 100,
         xgb.feature_importances_[-7:-3].sum() * 100,
         xgb.feature_importances_[-3:].sum() * 100],
        labels=['Bag of Words', 'Slice features', 'Shape features'],
        autopct='%1.2f%%', shadow=True, startangle=180)
plt.title('Features relevance according to classificator')
plt.show()

# %%
pca = PCA(n_components=2)
toplot = pca.fit_transform(np.concatenate((X_train, X_test), axis=0))
toplot_t = np.concatenate((y_train, y_test))

x0_min, x0_max = np.round(toplot.min()) - .2, np.round(toplot.max()) + .5
x1_min, x1_max = np.round(toplot.min()) + .2, np.round(toplot.max()) + .2

H = 0.01  # mesh stepsize

x0_axis_range = np.arange(x0_min, x0_max, H)
x1_axis_range = np.arange(x1_min, x1_max, H)

xx0, xx1 = np.meshgrid(x0_axis_range, x1_axis_range)
plt.scatter(xx0, xx1, s=0.5)

points = np.array([[xx0[i, j], xx1[i, j]] for i in range(xx0.shape[0]) for j in range(xx0.shape[1])])

pred = xgb.predict(pca.inverse_transform(points))

plt.scatter(x=points[:, 0], y=points[:, 1], c=pred, cmap='prism')

plt.scatter(x=toplot[:, 0][toplot_t == 0], y=toplot[:, 1][toplot_t == 0], c='red', marker='s', edgecolors='black')
plt.scatter(x=toplot[:, 0][toplot_t == 1], y=toplot[:, 1][toplot_t == 1], c='orange', marker='o', edgecolors='black')
plt.scatter(x=toplot[:, 0][toplot_t == 2], y=toplot[:, 1][toplot_t == 2], c='green', marker='^', edgecolors='black')
plt.title('decision regions of classifier for two pca component')

plt.xlim(points[:, 0].min(), points[:, 0].max())
plt.ylim(points[:, 1].min(), points[:, 1].max())
plt.xlabel('pca-1')
plt.ylabel('pca-2')

plt.show()

# %%
descs = pd.DataFrame(np.concatenate((X_train, X_test), axis=0))
descs.columns = [f"k_{i}" for i in range(50)] + list(shape_df.names[2:])
descs['img'] = np.concatenate((idx_train, idx_test))

pid = pd.read_csv('data/pid.csv')
del pid['Unnamed: 0']

descs = pd.merge(descs, features.to_pandas(), on='img')
descs = pd.merge(descs, pid, on='img')
descs.to_csv('data/img_descriptors.csv')