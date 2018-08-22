import tensorflow as tf
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sys
from gru import GRUCell

def variable_summaries(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)


def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    length = tf.cast(length, tf.int32)
    return length


def dense(inputs, activation, units, initializer=tf.contrib.layers.xavier_initializer(), name='dense'):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('weights', [shape[1], units], tf.float32, initializer)
        b = tf.get_variable('bias', [units], tf.float32, tf.constant_initializer(0.0))

        out = tf.nn.bias_add(tf.matmul(inputs, w), b)

    if activation != None:
        return activation(out), w, b
    else:
        return out, w, b


def gru(n_hidden, keep_prob, use_layernorm, initializer=tf.contrib.layers.xavier_initializer(), activation=tf.nn.relu, name='gru'):
    if use_layernorm:
        gru = GRUCell(n_hidden, activation=activation, kernel_initializer=initializer, name=name)
    else:
        gru = tf.nn.rnn_cell.GRUCell(n_hidden, kernel_initializer=initializer, activation=activation, name=name)
    drop = tf.nn.rnn_cell.DropoutWrapper(gru, output_keep_prob=keep_prob)
    return drop


def hierarchical_attention(input, attention_size, n_hidden):
    # attention mechanism as described in http://www.aclweb.org/anthology/N16-1174
    with tf.name_scope('Attention_layer'):
        # Trainable parameters
        w_omega = tf.Variable(tf.random_normal([n_hidden*2, attention_size], stddev=0.1))  # bidirectional --> 2 times the hidden state size
        b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
        u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

        with tf.name_scope('v'):
            # Applying fully connected layer with non-linear activation to each of the B*T timesteps
            # shape of v is (B,T,D)*(D,A)=(B,T,A), where B=batch_size, T=timesteps, A=attention_size, D=n_hidden*2
            v = tf.tanh(tf.tensordot(input, w_omega, axes=1) + b_omega)

        # For each of the timesteps its vector of size A from v is reduced with u
        vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
        alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

        # sum((B,D) * alpha_i), i=0..T, where alpha=attention vector
        attention_output = tf.reduce_sum(input * tf.expand_dims(alphas, -1), 1)

    return attention_output


def attention(input, n_hidden):
    # attention mechanism as described in https://arxiv.org/pdf/1409.0473.pdf (adjusted for sequence classificaion)
    with tf.name_scope('Attention_layer'):
        w = tf.Variable(tf.random_normal([n_hidden*2, 1], stddev=0.1))
        b = tf.Variable(tf.random_normal([1], stddev=0.1))
        e = tf.nn.bias_add(tf.tensordot(input, w, axes=1), b)  # shape (B,T,D)*(D,1)=(B, T, 1)
        alphas = tf.nn.softmax(e, axis=1)  # shape (B, T, 1)
        attention_output = tf.reduce_sum(tf.multiply(input, alphas), 1)  # shape (B, D)

    return attention_output


def add_features(sequence):
    sequence = np.asarray(sequence)
    next_seq = np.append(sequence[1:, :], [sequence[-1, :]], axis=0)
    prev_seq = np.append([sequence[0, :]], sequence[:-1, :], axis=0)

    # compute gradient
    gradient = np.subtract(sequence, prev_seq)

    #compute curvature
    vec_1 = np.multiply(gradient, -1)
    vec_2 = np.subtract(next_seq, sequence)
    angle = np.divide(np.sum(vec_1*vec_2, axis=1),
                      np.linalg.norm(vec_1, 2, axis=1)*np.linalg.norm(vec_2, 2, axis=1))
    curvature = np.column_stack((np.cos(angle), np.sin(angle)))

    #compute vicinity (5-points) - curliness/linearity
    padded_seq = np.concatenate(([sequence[0]], [sequence[0]], sequence, [sequence[-1]], [sequence[-1]]), axis=0)
    aspect = np.zeros(len(sequence))
    slope = np.zeros((len(sequence), 2))
    curliness = np.zeros(len(sequence))
    linearity = np.zeros(len(sequence))
    for j in range(2, len(sequence)+2):
        vicinity = np.asarray([padded_seq[j-2], padded_seq[j-1], padded_seq[j], padded_seq[j+1], padded_seq[j+2]])
        delta_x = max(vicinity[:, 0]) - min(vicinity[:, 0])
        delta_y = max(vicinity[:, 1]) - min(vicinity[:, 1])
        slope_vec = vicinity[-1] - vicinity[0]

        #aspect of trajectory
        aspect[j-2] = (delta_y - delta_x) / (delta_y + delta_x)

        #cos and sin of slope_angle of straight line from vicinity[0] to vicinity[-1]
        slope_angle = np.arctan(np.abs(np.divide(slope_vec[1], slope_vec[0]))) * np.sign(np.divide(slope_vec[1], slope_vec[0]))
        slope[j-2] = [np.cos(slope_angle), np.sin(slope_angle)]

        #length of trajectory divided by max(delta_x, delta_y)
        curliness[j-2] = np.sum([np.linalg.norm(vicinity[k+1] - vicinity[k], 2) for k in range(len(vicinity)-1)]) / max(delta_x, delta_y)

        #avg squared distance from each point to straight line from vicinity[0] to vicinity[-1]
        linearity[j-2] = np.mean([np.power(np.divide(np.cross(slope_vec, vicinity[0] - point), np.linalg.norm(slope_vec, 1)), 2) for point in vicinity])

    vicinity_features = np.column_stack((aspect, slope, curliness, linearity))

    # add features to signal
    result = np.nan_to_num(np.concatenate((sequence, gradient, curvature, vicinity_features), axis=1)).tolist()

    return result


def augment_data(X, y, iterations=0):
    for i in range(len(X)):
        for _ in range(iterations):
            current_s = X[i].tolist()

            # random up/down sample
            current_s = sample(current_s)

            current_s = np.asarray(current_s)
            pen_up = current_s[:, 2]
            current_s = current_s[:, 0:2]

            # random reverse
            current_s = reverse(current_s)

            # random shift
            current_s = shift(current_s)

            #random clinch/stretch
            current_s = stretch(current_s)

            #random rotate
            current_s = rotate(current_s)

            current_s = np.column_stack((current_s, pen_up)).tolist()

            X.append(current_s)
            y.append(y[i])

            #signal = np.asarray(X[i])
            #plt.plot(signal[:, 1], signal[:, 0])
            #plt.plot(np.asarray(current_s)[:, 1], np.asarray(current_s)[:, 0])
            #plt.show()

    return X, y


def reverse(sequence):
    if np.random.rand() > 0.7:
        sequence = np.flip(np.asarray(sequence), axis=0).tolist()
    return sequence


def stretch(sequence):
    for dim in range(len(sequence[0])):
        rand_stretch_factor = np.random.rand() * 0.8 - 0.4  # random factor [-0.4, 0.4]
        mean_value = np.mean(np.asarray(sequence)[:, dim])
        for i in range(len(sequence)):
            if sequence[i][dim] >= mean_value:
                sequence[i][dim] = sequence[i][dim] + (sequence[i][dim] - mean_value) * rand_stretch_factor
            else:
                sequence[i][dim] = sequence[i][dim] + (sequence[i][dim] - mean_value) * rand_stretch_factor

    return sequence


def shift(sequence):
    rand_offset = [np.random.rand() / 5, np.random.rand() / 5]
    sequence = np.asarray(sequence)
    if np.random.choice(2):
        # add offset
        sequence += rand_offset
    else:
        # subtract offset
        sequence -= rand_offset
    return sequence.tolist()


def rotate(sequence):
    theta = np.radians(np.random.randint(0, 22))  # random rotation angle theta [0, 22]
    c, s = np.cos(theta), np.sin(theta)
    rotation_mat = np.array(((c, -s), (s, c)))
    sequence = [np.matmul(rotation_mat, point).tolist() for point in sequence]
    return sequence


def sample(sequence):
    rand_sampling_rate = np.random.randint(5, 10)
    if np.random.choice(2):
        # downsample
        strokes_idx = [0]
        strokes_idx += [i for i in range(len(sequence)) if sequence[i] == [0., 0., 1.]]
        for i in reversed(range(1, len(strokes_idx))):
            if strokes_idx[i] - strokes_idx[i-1] >= rand_sampling_rate:
                for j in reversed(range(strokes_idx[i-1]+1, strokes_idx[i], rand_sampling_rate)):
                    sequence.remove(sequence[j])
    else:
        # upsample
        for i in range(0, len(sequence), rand_sampling_rate):
            if sequence[i] != [0., 0., 1.] and sequence[i + 1] != [0., 0., 1.]:
                point = np.mean([sequence[i + 1], sequence[i]], axis=0)
                point[2] = 0.
                sequence.insert(i + 1, point)
    return sequence


def get_current_input(data):
    sample_idx = []
    for i in range(len(data)):
        if "Touch Up" in data[i]:
            sample_idx.append(i)
    X = []
    current_idx = 0
    for i in sample_idx:
        curr_sample = data[current_idx:i]
        entry = []
        for line in curr_sample:
            values = line.replace("Touch Down: ", "")[:-2]
            point_x, point_y = values.split(" ")
            entry.append([float(point_x), float(point_y), 0.])
        current_idx = i+1
        entry.append([0., 0., 1.])
        X.append(entry)

    return X


def plot_confusion_matrix(cm, vocabulary,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = []
    for i in range(len(vocabulary)+1):
        if i == len(vocabulary):
            char = 'None'
        else:
            char = vocabulary[i]
        classes.append(char)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.rcParams.update({'font.size': 16})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.rcParams.update({'font.size': 18})
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.rcParams.update({'font.size': 16})

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.rcParams.update({'font.size': 12})
    plt.show()


class Logger(object):
    def __init__(self, logpath):
        self.terminal = sys.stdout
        self.log = open(logpath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        pass

    def close(self):
        self.log.close()
