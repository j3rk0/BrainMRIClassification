import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

df = pd.read_csv('data/img_descriptors.csv', index_col=0)

patient_data = {
    pid: {
        'descriptors': df.loc[df['pid'] == pid, 'k_0':'slice_center_y'],
        'label': df.loc[df['pid'] == pid, 'label'].unique()[0] - 1,
        'imgs': df.loc[df['pid'] == pid, 'img']
    } for pid in df['pid'].unique()
}

model = XGBClassifier()
model.load_model('models/booster.json')

pred = [model.predict(patient_data[x]['descriptors']) for x in patient_data.keys()]
pred = pd.DataFrame.from_dict({
    '0': [np.mean(x == 0) for x in pred],
    '1': [np.mean(x == 1) for x in pred],
    '2': [np.mean(x == 2) for x in pred]
})
y = np.array([patient_data[x]['label'] for x in patient_data.keys()])

correct = np.argmax(np.array(pred), axis=1) == y

print(f"accuracy: {np.mean(correct)}\nerrors:")

incorrect = pred.iloc[~correct, :]
print(incorrect)
conf = np.max(np.array(pred), axis=1)
print(f"\nsample under conf. treshold: {conf[conf < .7].shape[0]}")

# %%
import cv2

p = np.argwhere(y != np.argmax(np.array(pred), axis=1)).flatten()
p = [list(patient_data.keys())[pid] for pid in p]

imgs = [cv2.imread(f"data/images/mri/{img}.png", 0) for img in patient_data[p[0]]['imgs']]
masks = [cv2.imread(f"data/images/mask/{img}.png", 0) for img in patient_data[p[0]]['imgs']]
# %%

from lib.preprocessing import Preprocessor

prep = Preprocessor(None, None)
out = []
for img, mask in zip(imgs, masks):
    prep.set_img(img)
    prep.set_mask(mask)
    prep.preprocessing_pipeline()
    out.append(prep.img)

# %%
from lib.descriptor import SiftKeypointsExtractor

sift = SiftKeypointsExtractor(None)
kp = []
desc = []
for img in out:
    sift.set_img(img)
    sift.extract_keyp()
    kp.append(sift.keyp_loc)
    descriptors = sift.keyp_desc
    # descriptors /= (descriptors.sum(axis=1, keepdims=True) + 1e-7)
    # descriptors = np.sqrt(descriptors)
    desc.append(descriptors)

# %%
import pickle as pkl

tf_idf = pkl.load(open('models/tf_idf_50.sk', 'rb'))
k_means = pkl.load(open('models/kmeans_50.sk', 'rb'))

clust = [k_means.predict(k) for k in desc]

bow = []
for cl in clust:
    curr = np.zeros(50)
    for i in cl:
        curr[i] += 1
    bow.append(curr)

desc_tfidf = [tf_idf.transform([k]).toarray() for k in bow]
desc_tfidf = [d / d.sum(axis=1, keepdims=True) for d in desc_tfidf]

relevances = model.feature_importances_[:50]

data_relevant = [[i for i in range(50) if d[0][i] > np.quantile(d[0], .95)] for d in desc_tfidf]
model_relevant = [[i for i in range(50) if relevances[i] > np.quantile(relevances, .95)] for d in desc_tfidf]

fig, axs = plt.subplots(3, 3)
for i in range(len(out)):
    axs[0, i].imshow(out[i])
    axs[0, i].scatter(kp[i][:, 0], kp[i][:, 1], s=5, c=clust[i], cmap='Set1')

for i in range(len(out)):
    axs[1, i].imshow(out[i])
    sel = []
    for j in clust[i]:
        sel.append(j in data_relevant[i])
    sel = np.array(sel)
    axs[1, i].scatter(kp[i][sel, 0], kp[i][sel, 1], c=clust[i][sel], s=5, cmap='Set1')

for i in range(len(out)):
    axs[2, i].imshow(out[i])
    sel = []
    for j in clust[i]:
        sel.append(j in model_relevant[i])
    sel = np.array(sel)
    axs[2, i].scatter(kp[i][sel, 0], kp[i][sel, 1], c=clust[i][sel], s=5, cmap='Set1')

plt.show()

# %% plot roc curve

fpr, tpr, _ = roc_curve(y == 0, pred['0'])
roc_auc = auc(fpr, tpr)
plt.title("meningioma roc curve")
plt.xlabel('fpr')
plt.ylabel('tpr')

plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.plot(fpr, tpr, color="darkorange", label=f"auc: {roc_auc:.4f}")
plt.legend(loc=4)
plt.show()

# %%
fpr, tpr, _ = roc_curve(y == 1, pred['1'])
roc_auc = auc(fpr, tpr)
plt.title("glioma roc curve")
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.plot(fpr, tpr, color="darkorange", label=f"auc: {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.legend(loc=4)
plt.show()
# %%
fpr, tpr, _ = roc_curve(y == 2, pred['2'])
roc_auc = auc(fpr, tpr)
plt.title("pituritary roc curve")
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.plot(fpr, tpr, color="darkorange", label=f"auc: {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.legend(loc=4)
plt.show()
# %%
