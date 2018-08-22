import tensorflow as tf
import numpy as np
from model import Model
import os
import threading
import matplotlib.pyplot as plt
from ops import augment_data, add_features, Logger, plot_confusion_matrix
import datetime
import time
import argparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from capture_data import DataObserver
from sklearn.decomposition import PCA
import pickle
import sys
import warnings
from cnn_model import CNNModel
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

##CONFIG
CONFIG = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('model_to_load', 'baseline_RNN+SWA/model-2018-08-21T17_25_43', 'name of model to load')
tf.app.flags.DEFINE_string('root_path', './data/', 'root path for train, validation and test data')
timestamp = datetime.datetime.now().isoformat().split('.')[0].replace(':', '_')
tf.app.flags.DEFINE_string('model_dir', './experiments/model-' + timestamp + '/', 'directory to store currently trained model')

# Data Parameters
tf.app.flags.DEFINE_bool('use_unipen_data', False, 'use data from unipen dataset for training')
tf.app.flags.DEFINE_bool('use_augmentation', True, 'random augmentation of the training data')
tf.app.flags.DEFINE_integer('augment_iter', 4, 'how many augmented copies of one sequence')
tf.app.flags.DEFINE_bool('use_normalization', True, 'data normalization with data_scaler')

# Training Parameters
tf.app.flags.DEFINE_integer('num_epochs', 10, 'number of epochs to run training')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch size for training')
tf.app.flags.DEFINE_integer('display_step', 50, 'display training statistics every ... steps in command line')
tf.app.flags.DEFINE_float('dropout_keep_prob', 0.9, 'keep probability for dropout')
tf.app.flags.DEFINE_bool('use_SWA', False, 'Stochastic Weight Averaging: average all models saved during training')  # https://arxiv.org/pdf/1803.05407.pdf
tf.app.flags.DEFINE_integer('save_step', 10, 'save model every ... epochs')

tf.app.flags.DEFINE_bool('use_cyclic_LR_schedule', False, 'cyclic learning rate schedule. Reset LR after each save_step')  # https://arxiv.org/pdf/1608.03983.pdf
tf.app.flags.DEFINE_float('lr_init', 0.0075, 'initial learning rate')
tf.app.flags.DEFINE_float('lr_min', 0.000075, 'minimum learning rate')
tf.app.flags.DEFINE_integer('lr_n_decrease', 10, 'how many times to decrease before lr should reach minimum')

vocabulary = 'PEAWSB'
data_scaler = StandardScaler()
#pca = PCA(n_components=n_features)

# Network Parameters
tf.app.flags.DEFINE_bool('use_attention', False, 'use attention mechanism')
tf.app.flags.DEFINE_bool('use_layernorm', False, 'use layer normalization for GRU cells')
tf.app.flags.DEFINE_integer('n_layers', 1, 'number of hidden layers')
tf.app.flags.DEFINE_integer('n_hidden', 24, 'number of units in hidden layers')
tf.app.flags.DEFINE_integer('n_classes', len(vocabulary)+1, 'number of classes')
tf.app.flags.DEFINE_integer('n_features', 12, 'RNN input dimensionality')
tf.app.flags.DEFINE_list('class_weighting', [1., 1., 1., 1., 1., 1., 3.], 'loss weighting for every class')


def prepare_data(pad_length=False):
    X = []
    y = []

    for i in range(CONFIG.n_classes):
        if i == CONFIG.n_classes-1:
            char = 'None'
        else:
            char = vocabulary[i]
        res_x = pickle.load(open(CONFIG.root_path + char + ".pkl", 'rb'))
        res_y = np.tile(np.eye(CONFIG.n_classes)[i], (len(res_x), 1)).tolist()
        X += res_x
        y += res_y

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    if CONFIG.use_augmentation:
        X_train, y_train = augment_data(X_train, y_train, iterations=CONFIG.augment_iter)

    # add features and normalize data to 0 mean and unit variance
    pen_up = []
    for i in range(len(X_train)):
        sequence = np.asarray(X_train[i])
        pen_up.append(sequence[:, 2])
        sequence = sequence[:, 0:2]
        sequence = add_features(sequence)
        if CONFIG.use_normalization:
            X_train[i] = sequence
        else:
            X_train[i] = np.column_stack((sequence, pen_up[i])).tolist()

    if CONFIG.use_normalization:
        global data_scaler
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
        if CONFIG.use_normalization:
            sequence = data_scaler.transform(sequence)
        X_test[i] = np.column_stack((sequence, pen_up)).tolist()

    # # dimensionality reduction with PCA
    # pca = PCA(n_components=8)
    # pca.fit(np.vstack(X_train))
    #
    # for i in range(len(X_train)):
    #     X_train[i] = pca.transform(X_train[i]).tolist()
    #
    # for i in range(len(X_test)):
    #     X_test[i] = pca.transform(X_test[i]).tolist()

    # plot pca result
    # fig, ax1 = plt.subplots()
    # ax1.plot()
    # ax1.set_xlabel('components')
    # ax1.set_ylabel('variance percentage')
    # ax1.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='tab:blue')
    # plt.show()

    if pad_length:
        max_seqLen = max(len(max(X_train, key=len)), len(max(X_test, key=len)))
        # Pad sequences for dimension consistency
        padding_mask = np.zeros(CONFIG.n_features).tolist()
        for i in range(len(X_train)):
            X_train[i] += [padding_mask for _ in range(max_seqLen - len(X_train[i]))]

        for i in range(len(X_test)):
            X_test[i] += [padding_mask for _ in range(max_seqLen - len(X_test[i]))]

    return X_train, X_test, y_train, y_test


def prepare_unipen_data():
    X = pickle.load(open(CONFIG.root_path + "unipen_X.pkl", 'rb'))
    y = pickle.load(open(CONFIG.root_path + "unipen_y.pkl", 'rb'))
    # avg_len = 0
    # for entry in X:
    #     avg_len += len(entry)
    # avg_len = avg_len / len(X)
    # print(avg_len)
    for i in reversed(range(len(y))):
        if np.argmax(y[i]) == 6 and np.random.rand() > 0.07:
            y.pop(i)
            X.pop(i)

    # class_counts = np.zeros(CONFIG.n_classes)
    # for entry in y:
    #     class_counts[np.argmax(entry)] += 1
    # print(class_counts)
    # global data_scaler
    # data_scaler = pickle.load(open(CONFIG.root_path + "unipen_scaler.pkl", 'rb'))
    return X, y


def get_batch(X, y, num_batch, size_batch):
    # Get indices
    batch_start_idx = int(num_batch*size_batch)
    batch_end_idx = int(num_batch*size_batch + size_batch)
    # Get batch data
    batch_x = X[batch_start_idx:batch_end_idx]
    batch_y = y[batch_start_idx:batch_end_idx]

    # Get max length of batch
    batch_seqLen = len(max(batch_x, key=len))
    # Pad sequences for dimension consistency
    padding_mask = np.zeros(CONFIG.n_features).tolist()
    for sequence in batch_x:
        sequence += [padding_mask for _ in range(batch_seqLen - len(sequence))]

    return np.asarray(batch_x), np.asarray(batch_y)


def train(sess, model, X, y, X_val, y_val):
    # Perform training
    print("Start training..")
    saver = tf.train.Saver(max_to_keep=int(CONFIG.num_epochs / CONFIG.save_step)+1)

    X_val, y_val = get_batch(X_val, y_val, 0, len(X_val))

    num_batches = int(len(X)/CONFIG.batch_size)
    if (len(X) % CONFIG.batch_size != 0):
        num_batches += 1

    for epoch in range(CONFIG.num_epochs):
        X, y = shuffle(X, y)
        for batch in range(num_batches):
            batch_x, batch_y = get_batch(X, y, batch, CONFIG.batch_size)

            # Run model on batch
            loss, accuracy = model.train(sess, batch_x, batch_y, CONFIG.dropout_keep_prob, epoch * num_batches + batch)

            if batch % CONFIG.display_step == 0 or batch == 0:
                val_loss, val_accuracy = model.validate(sess, X_val, y_val, epoch * num_batches + batch)
                print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss= " +
                      "{:.6f}".format(loss) + ", Training Accuracy= " +
                      "{:.5f}".format(accuracy) + ", Validation Loss= " +
                      "{:.5f}".format(val_loss) + ", Validation Accuracy= " +
                      "{:.5f}".format(val_accuracy))

        if (epoch+1) % CONFIG.save_step == 0 and epoch != 0:
            saver.save(sess, CONFIG.model_dir + 'model-epoch-' + str(epoch+1) + '.cptk')
            print("Model saved.")

    if CONFIG.use_SWA:
        # create average of all saved models
        print("Averaging models..")
        variable_collection = tf.trainable_variables()
        n_models = 1
        for checkpoint in range(CONFIG.save_step, CONFIG.num_epochs, CONFIG.save_step):
            saver.restore(sess, CONFIG.model_dir + 'model-epoch-' + str(checkpoint) + '.cptk')
            for new_variable in tf.trainable_variables():
                value_old_model = 0
                for old_variable in variable_collection:
                    if new_variable.name == old_variable.name:
                        value_old_model = sess.run(old_variable)

                value_curr_model = sess.run(new_variable)
                sess.run(new_variable.assign(tf.divide(tf.add(tf.multiply(value_curr_model, n_models), value_old_model), n_models + 1)))
            n_models += 1

    saver.save(sess, CONFIG.model_dir + 'model-epoch-final.cptk')
    print("Training done, final model saved")


def test(sess, model, X, y):
    X, y = get_batch(X, y, 0, len(X))
    test_loss, test_accuracy, test_confusion_matrix = model.test(sess, X, y)
    print("Test loss: ", test_loss)
    print("Test Accuracy: ", test_accuracy)
    print("Test Confusion Matrix: ")
    print(test_confusion_matrix)
    #plot_confusion_matrix(test_confusion_matrix, vocabulary)


def predict(sess, model):
    # prediction for sample
    observer = DataObserver("demo.log")
    print("Real-time prediction started.")
    while True:
        new_entry = observer.step()
        if new_entry != None:
            sequence = np.asarray(new_entry)
            pen_up = sequence[:, 2]
            sequence = sequence[:, 0:2]
            sequence = add_features(sequence)
            if CONFIG.use_normalization:
                sequence = data_scaler.transform(sequence)
            sequence = np.column_stack((sequence, pen_up)).tolist()

            #get my prediction
            output = model.predict(sess, [sequence])
            prediction = np.argmax(output[0], axis=0)

            if prediction < len(vocabulary):
                print("Input detected: " + str(vocabulary[prediction]) + ", Probabilities: " + str(output[0]))

        time.sleep(0.05)


def launchTensorBoard():
    import os
    os.system('tensorboard --logdir=' + CONFIG.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', help='boolean for loading a model', dest='load_model', type=bool, default=False)
    parser.add_argument('-d', help='directory to load the model', dest='load_dir', type=str,
                        default='./experiments/')
    args, unknown = parser.parse_known_args()
    sys.argv[1:] = unknown

    load_dir = os.path.join(args.load_dir, CONFIG.model_to_load)
    if args.load_model:
        t = threading.Thread(target=launchTensorBoard, args=([]))
        CONFIG.model_dir = load_dir + '/'
        t.start()
    else:
        old_stdout = sys.stdout
        os.mkdir(CONFIG.model_dir)
        logger = Logger(CONFIG.model_dir + "training.log")
        sys.stdout = logger
        print("Model parameters: ", tf.app.flags.FLAGS.flag_values_dict())

    if CONFIG.use_unipen_data:
        X, y = prepare_unipen_data()
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
        X_test, _, y_test, _ = prepare_data(pad_length=False)
    else:
        X_train, X_test, y_train, y_test = prepare_data(pad_length=False)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.75, random_state=42, stratify=y_test)

    # determine learning rate schedule
    num_batches = int(len(X_train) / CONFIG.batch_size)
    if (len(X_train) % CONFIG.batch_size != 0):
        num_batches += 1
    lr_decay = np.power((CONFIG.lr_min / CONFIG.lr_init), (1 / CONFIG.lr_n_decrease))
    lr_step = int(CONFIG.num_epochs * num_batches / CONFIG.lr_n_decrease)

    # create model
    sess = tf.Session()
    model = Model(sess, CONFIG, lr_decay, lr_step)
    #model = CNNModel(sess, CONFIG, lr_decay, lr_step, len(X_train[0]))  # pad_length must be True

    # Initialize the variables (i.e. assign their default value)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # output number of parameters for visualizing model complexity
    total_parameters = 0
    print("Number of parameters by variable:")
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = np.prod(shape.as_list())
        print(variable.name + " " + str(shape) + ": " + str(variable_parameters))
        total_parameters += variable_parameters
    print("Total number of model parameters: " + str(total_parameters))

    if not args.load_model:
        # start tensorboard visualization of the learning process
        t = threading.Thread(target=launchTensorBoard, args=([]))
        t.start()

        # train model
        train(sess, model, X_train, y_train, X_val, y_val)
    else:
        restore_model = tf.train.Saver()
        try:
            restore_model.restore(sess, os.path.join(load_dir, "model-epoch-final.cptk"))
            print("Model restored.")
        except Exception as e:
            print("Model not restored: ", str(e))
            exit(0)

    # evaluate model
    test(sess, model, X_test, y_test)

    if not args.load_model:
        sys.stdout = old_stdout
        logger.close()

    predict(sess, model)


