import tensorflow as tf
import numpy as np
from ops import dense, variable_summaries
import tensorflow.contrib.slim as slim

class CNNModel:
    def __init__(self, sess, CONFIG, lr_decay, lr_step, seq_maxlen):
        self.n_hidden = CONFIG.n_hidden
        self.n_layers = CONFIG.n_layers
        self.n_classes = CONFIG.n_classes
        self.model_dir = CONFIG.model_dir

        self.lr_init = CONFIG.lr_init
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.lr_min = CONFIG.lr_min
        self.n_features = CONFIG.n_features
        self.seq_maxlen = seq_maxlen

        # Input params
        self.input = tf.placeholder('float32', [None, self.seq_maxlen, self.n_features, 1])  # [batch_size, seq_length, n_features]
        self.labels = tf.placeholder('float32', [None, self.n_classes])
        self.keep_prob = tf.placeholder('float32')
        self.global_step = tf.Variable(0, trainable=False)

        with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            with tf.variable_scope('Convolution'):
                net = slim.conv2d(self.input, 16, [1, 3], padding='VALID', scope='conv1')
                net = slim.conv2d(net, 16, [1, 3], padding='VALID', scope='conv2')
                net = slim.max_pool2d(net, [3, 1], stride=3, scope='pool1')
                net = slim.conv2d(net, 32, [1, 2], padding='VALID', scope='conv3')
                net = slim.conv2d(net, 32, [1, 2], padding='VALID', scope='conv4')
                net = slim.max_pool2d(net, [3, 1], stride=3, scope='pool2')
                net = slim.dropout(net, self.keep_prob, scope='Dropout')

        flat = tf.contrib.layers.flatten(net)
        dense_out, dense_w, dense_b = dense(flat, tf.nn.relu, 50, name='dense')
        output = tf.nn.dropout(dense_out, self.keep_prob)

        self.logits, self.output_W, self.output_b = dense(output, None, self.n_classes, name='softmax')
        self.logits = tf.reshape(self.logits, [tf.shape(self.input)[0], self.n_classes])

        # class weights
        class_weights = tf.constant([CONFIG.class_weighting])
        weight_per_label = tf.matmul(self.labels, tf.transpose(class_weights))

        # compute mean weighted loss
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
        self.mean_loss = tf.reduce_mean(tf.multiply(weight_per_label, tf.expand_dims(losses, axis=-1)))

        # run optimization
        self.learning_rate_op = tf.maximum(self.lr_min,
                                           tf.train.exponential_decay(
                                               self.lr_init,
                                               self.global_step,
                                               self.lr_step,
                                               self.lr_decay,
                                               staircase=True))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate_op)
        self.gradients_vars = self.optimizer.compute_gradients(self.mean_loss)
        self.clipped_gradients_vars = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gradients_vars]
        self.train_op = self.optimizer.apply_gradients(self.clipped_gradients_vars)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.constructSummary(sess)

    def constructSummary(self, sess):
        variable_summaries(self.mean_loss, 'loss')
        variable_summaries(self.accuracy, 'accuracy')
        variable_summaries(self.learning_rate_op, 'learning_rate')
        variable_summaries(self.output_W, 'output_weights')
        variable_summaries(self.gradients_vars[0], 'gradients')
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.model_dir + 'train', sess.graph)
        self.val_writer = tf.summary.FileWriter(self.model_dir + 'validation', sess.graph)

    def train(self, sess, batch_input, batch_labels, keep_prob, global_step):

        _, loss, label_err_rate, statistics = sess.run([self.train_op, self.mean_loss, self.accuracy, self.merged],
                                                       feed_dict={self.input: np.expand_dims(batch_input, axis=-1),
                                                            self.labels: batch_labels,
                                                            self.global_step: global_step,
                                                            self.keep_prob: keep_prob})

        self.train_writer.add_summary(statistics, global_step)
        return loss, label_err_rate

    def validate(self, sess, test_input, test_labels, global_step):
        loss, label_err_rate, statistics = sess.run([self.mean_loss, self.accuracy, self.merged],
                                                    feed_dict={self.input: np.expand_dims(test_input, axis=-1),
                                                         self.labels: test_labels,
                                                         self.keep_prob: 1.0})
        self.val_writer.add_summary(statistics, global_step)
        return loss, label_err_rate

    def test(self, sess, test_input, test_labels):

        loss, accuracy, logits = sess.run([self.mean_loss, self.accuracy, self.logits],
                                                feed_dict={self.input: np.expand_dims(test_input, axis=-1),
                                                          self.labels: test_labels,
                                                          self.keep_prob: 1.0})
        confusion_matrix = tf.confusion_matrix(tf.argmax(logits, axis=1),
                                               tf.argmax(test_labels, axis=1)).eval(session=sess)

        return loss, accuracy, confusion_matrix

    def predict(self, sess, pred_input):
        pred_input = pred_input[0].tolist()
        padding_mask = np.zeros(self.n_features).tolist()
        pred_input += [padding_mask for _ in range(self.seq_maxlen - len(pred_input))]
        pred_input = np.expand_dims(np.asarray([pred_input]), axis=-1)
        logits = sess.run(self.logits, feed_dict={self.input: pred_input,
                                                  self.keep_prob: 1.0})
        return logits

