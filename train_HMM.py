import numpy as np
import os
import datetime
from hmmlearn.hmm import GMMHMM
import sklearn.metrics
from ops import augment_data, add_features
from sklearn.preprocessing import MinMaxScaler
import warnings
import time
import pickle
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

##CONFIG
root_path = './data/'
timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
model_dir = './experiments/model-' + timestamp + '/'

# Parameters
vocabulary = 'PEAWSB'
n_classes = len(vocabulary)+1  # number of classes
data_scaler = MinMaxScaler(feature_range=(0, 1))


def prepare_data(augment_iter=0):
    X = []
    y = []

    for i in range(n_classes):
        if i == n_classes - 1:
            char = 'None'
        else:
            char = vocabulary[i]
        res_x = pickle.load(open(root_path + char + ".pkl", 'rb'))
        res_y = np.tile(i, (len(res_x), 1)).tolist()
        X += res_x
        y += res_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    X_train, y_train = augment_data(X_train, y_train, iterations=augment_iter)

    # add features and normalize data
    pen_up = []
    for i in range(len(X_train)):
        sequence = np.asarray(X_train[i])
        pen_up.append(sequence[:, 2])
        sequence = sequence[:, 0:2]
        sequence = add_features(sequence)
        X_train[i] = sequence

    data_scaler.fit(np.vstack(X_train))
    for i in range(len(X_train)):
        sequence = np.asarray(X_train[i])
        sequence = data_scaler.transform(sequence)
        X_train[i] = np.column_stack((sequence, pen_up[i]))

    for i in range(len(X_test)):
        sequence = np.asarray(X_test[i])
        pen_up = sequence[:, 2]
        sequence = sequence[:, 0:2]
        sequence = add_features(sequence)
        sequence = data_scaler.transform(sequence)
        X_test[i] = np.column_stack((sequence, pen_up))

    return X_train, X_test, y_train, y_test


def train(models, X, y):
    # Perform training
    print("Start training..")

    #create subsets for training
    X_dict = {}
    for i in range(n_classes):
        X_dict.update({i: ([], [])})

    for sequence, label in list(zip(X, y)):
        X_curr, len_curr = X_dict.get(label[0])
        X_curr.append(sequence)
        len_curr.append(len(sequence))
        X_dict.update({label[0]: (X_curr, len_curr)})

    #model training
    for key, model in models.items():
        X_curr, len_curr = X_dict.get(key)
        model.fit(np.vstack(X_curr), len_curr)

    #saver.save(sess, model_dir + 'model.cptk')
    print("Training done, final model saved")


def test(models, X, y):
    # prediction sample for every entry of test set
    prediction = np.zeros(len(X))

    for i in range(len(X)):
        prediction[i] = np.argmax([model.score(X[i]) for key, model in models.items()])

    test_confusion_matrix = sklearn.metrics.confusion_matrix(y, prediction, labels=range(n_classes))
    test_accuracy = np.sum(np.diagonal(test_confusion_matrix)) / np.sum(test_confusion_matrix)

    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data(augment_iter=4)

    # Test Accuracy:  0.7905
    models = {}
    for i in range(n_classes):
        models.update({i: GMMHMM(n_components=8, n_mix=3)})

    # train models
    train(models, X_train, y_train)

    # evaluate models
    test(models, X_test, y_test)

