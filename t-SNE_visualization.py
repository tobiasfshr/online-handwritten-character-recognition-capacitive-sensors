from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pickle

tsne = TSNE(n_components=2, init='pca', random_state=0)

vocabulary = 'PEAWSB'
root_path = './data/'
n_features = 3
n_classes = len(vocabulary)+1


def prepare_data():
    X = []
    y = []

    for i in range(n_classes):
        if i == n_classes - 1:
            char = 'None'
        else:
            char = vocabulary[i]
        res_x = pickle.load(open(root_path + char + ".pkl", 'rb'))
        res_y = np.tile(i, len(res_x)).tolist()
        X += res_x
        y += res_y

    max_seqLen = len(max(X, key=len))
    # Pad sequences for dimension consistency
    padding_mask = np.zeros(n_features).tolist()
    for i in range(len(X)):
        X[i] = X[i].tolist()
        X[i] += [padding_mask for _ in range(max_seqLen - len(X[i]))]


    # flat sequence
    X = np.asarray(X)
    shape = np.shape(X)
    X = np.reshape(X, (shape[0], shape[1] * shape[2]))

    return X, y


X, y = prepare_data()
tsne_results = tsne.fit_transform(X)

plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=y, cmap=plt.cm.get_cmap("gist_rainbow", n_classes))
plt.colorbar(ticks=range(n_classes))
plt.title("t-SNE")
plt.axis('tight')
plt.show()
