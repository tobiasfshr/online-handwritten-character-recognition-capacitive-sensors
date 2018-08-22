import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import sklearn.metrics
from ops import add_features, augment_data
from sklearn.model_selection import train_test_split
from capture_data import DataObserver
from sklearn.preprocessing import StandardScaler
import warnings
import time
import pickle
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

##CONFIG
root_path = './data/'
timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
model_dir = './experiments/model-' + timestamp + '/'

# Parameters
vocabulary = 'PEAWSB'
n_classes = len(vocabulary)+1  # number of classes
data_scaler = StandardScaler()
n_features = 12


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

    # add features and normalize data to 0 mean and unit variance
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
        X_train[i] = np.column_stack((sequence, pen_up[i])).tolist()

    for i in range(len(X_test)):
        sequence = np.asarray(X_test[i])
        pen_up = sequence[:, 2]
        sequence = sequence[:, 0:2]
        sequence = add_features(sequence)
        sequence = data_scaler.transform(sequence)
        X_test[i] = np.column_stack((sequence, pen_up)).tolist()

    max_seqLen = max(len(max(X_train, key=len)), len(max(X_test, key=len)))
    # Pad sequences for dimension consistency
    padding_mask = np.zeros(n_features).tolist()
    for i in range(len(X_train)):
        X_train[i] += [padding_mask for _ in range(max_seqLen - len(X_train[i]))]

    for i in range(len(X_test)):
        X_test[i] += [padding_mask for _ in range(max_seqLen - len(X_test[i]))]

    # flat sequence
    X_train = np.asarray(X_train)
    shape = np.shape(X_train)
    X_train = np.reshape(X_train, (shape[0], shape[1] * shape[2]))

    X_test = np.asarray(X_test)
    shape = np.shape(X_test)
    X_test = np.reshape(X_test, (shape[0], shape[1] * shape[2]))

    return X_train, X_test, y_train, y_test


def train(model, X, y):
    # Perform training
    print("Start training..")

    #model_training
    model.fit(X, y)

    #saver.save(sess, model_dir + 'model.cptk')
    print("Training done, final model saved")


def test(model, X, y):
    # prediction sample for every entry of test set
    prediction = model.predict(X)

    test_confusion_matrix = sklearn.metrics.confusion_matrix(y, prediction, labels=range(n_classes))
    test_accuracy = np.sum(np.diagonal(test_confusion_matrix)) / np.sum(test_confusion_matrix)

    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix:")
    print(test_confusion_matrix)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data(augment_iter=4)

    #model = RandomForestClassifier(n_estimators=20, max_depth=12)  # Test Accuracy:  0.7238
    #model = SVC()  # linear: Test Accuracy: 0.7428, rbf: 0.7142
    model = MLPClassifier(hidden_layer_sizes=(300,), learning_rate='adaptive', random_state=1)  # Test Accuracy: 0.8
    # train models
    train(model, X_train, y_train)

    # evaluate models
    test(model, X_test, y_test)

