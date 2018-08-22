import tensorflow as tf
import numpy as np
from ops import dense, length, gru, variable_summaries, attention, hierarchical_attention


class Model:
    def __init__(self, sess, CONFIG, lr_decay, lr_step):
        self.n_hidden = CONFIG.n_hidden
        self.n_layers = CONFIG.n_layers
        self.n_classes = CONFIG.n_classes
        self.model_dir = CONFIG.model_dir

        self.lr_init = CONFIG.lr_init
        self.lr_decay = lr_decay
        self.lr_step = lr_step
        self.lr_min = CONFIG.lr_min

        # Input params
        self.input = tf.placeholder('float32', [None, None, CONFIG.n_features])  # [batch_size, seq_length, ..]
        self.labels = tf.placeholder('float32', [None, self.n_classes])
        self.initial_state_fw = tf.placeholder(tf.float32, [self.n_layers, None, self.n_hidden])
        self.initial_state_bw = tf.placeholder(tf.float32, [self.n_layers, None, self.n_hidden])
        self.keep_prob = tf.placeholder('float32')
        self.global_step = tf.Variable(0, trainable=False)

        # init RNN cells
        self.stacked_cells_fw = [gru(self.n_hidden, self.keep_prob, CONFIG.use_layernorm, name="fw_gru-"+str(n)) for n in range(self.n_layers)]
        self.stacked_cells_bw = [gru(self.n_hidden, self.keep_prob, CONFIG.use_layernorm, name="bw_gru-"+str(n)) for n in range(self.n_layers)]

        initial_state_tuple_fw = tuple(self.initial_state_fw[idx] for idx in range(self.n_layers))
        initial_state_tuple_bw = tuple(self.initial_state_bw[idx] for idx in range(self.n_layers))

        # Get gru cell output
        curr_input = self.input
        seqlen = length(self.input)
        for n in range(self.n_layers):
            cell_fw = self.stacked_cells_fw[n]
            cell_bw = self.stacked_cells_bw[n]
            init_state_fw = initial_state_tuple_fw[n]
            init_state_bw = initial_state_tuple_bw[n]
            states_series, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, curr_input,
                                                                               initial_state_fw=init_state_fw,
                                                                               initial_state_bw=init_state_bw,
                                                                               dtype=tf.float32, sequence_length=seqlen)
            curr_input = tf.concat(states_series, 2)

        rnn_output = tf.concat([final_states[0], final_states[1]], 1)

        if CONFIG.use_attention:
            rnn_output = attention(curr_input, self.n_hidden)

        # create output logits for efficient softmax computation and prediction
        self.logits, self.output_W, self.output_b = dense(rnn_output, None, self.n_classes, name='softmax')
        self.logits = tf.reshape(self.logits, [tf.shape(self.input)[0], self.n_classes])

        # class weights
        class_weights = tf.constant([CONFIG.class_weighting])
        weight_per_label = tf.matmul(self.labels, tf.transpose(class_weights))

        # compute mean weighted loss
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
        self.mean_loss = tf.reduce_mean(tf.multiply(weight_per_label, tf.expand_dims(losses, axis=-1)))

        #compute current learning rate based on corresponding schedule
        if CONFIG.use_cyclic_LR_schedule:
            n_cycles = int(CONFIG.num_epochs / CONFIG.save_step)
            lr_step_cycle = int(self.lr_step / n_cycles)
            cycle_step = tf.mod(self.global_step, lr_step_cycle * CONFIG.lr_n_decrease)
            self.learning_rate_op = tf.maximum(self.lr_min, tf.train.exponential_decay(self.lr_init, cycle_step,
                                                                                       lr_step_cycle, self.lr_decay,
                                                                                       staircase=True))
        else:
            self.learning_rate_op = tf.maximum(self.lr_min, tf.train.exponential_decay(self.lr_init, self.global_step,
                                                                                       self.lr_step, self.lr_decay,
                                                                                       staircase=True))

        # run optimization
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
        # create initial zero state
        initial_state_fw = np.zeros((self.n_layers, len(batch_input), self.n_hidden))
        initial_state_bw = np.zeros((self.n_layers, len(batch_input), self.n_hidden))

        _, loss, label_err_rate, statistics = sess.run([self.train_op, self.mean_loss, self.accuracy, self.merged],
                                                       feed_dict={self.input: batch_input,
                                                            self.labels: batch_labels,
                                                            self.initial_state_fw: initial_state_fw,
                                                            self.initial_state_bw: initial_state_bw,
                                                            self.global_step: global_step,
                                                            self.keep_prob: keep_prob})

        self.train_writer.add_summary(statistics, global_step)
        return loss, label_err_rate

    def validate(self, sess, test_input, test_labels, global_step):
        # create initial zero state
        initial_state_fw = np.zeros((self.n_layers, len(test_input), self.n_hidden))
        initial_state_bw = np.zeros((self.n_layers, len(test_input), self.n_hidden))
        loss, label_err_rate, statistics = sess.run([self.mean_loss, self.accuracy, self.merged],
                                                    feed_dict={self.input: test_input,
                                                         self.labels: test_labels,
                                                         self.initial_state_fw: initial_state_fw,
                                                         self.initial_state_bw: initial_state_bw,
                                                         self.keep_prob: 1.0})
        self.val_writer.add_summary(statistics, global_step)
        return loss, label_err_rate

    def test(self, sess, test_input, test_labels):
        # create initial zero state
        initial_state_fw = np.zeros((self.n_layers, len(test_input), self.n_hidden))
        initial_state_bw = np.zeros((self.n_layers, len(test_input), self.n_hidden))
        loss, accuracy, logits = sess.run([self.mean_loss, self.accuracy, self.logits],
                                                feed_dict={self.input: test_input,
                                                          self.labels: test_labels,
                                                          self.initial_state_fw: initial_state_fw,
                                                          self.initial_state_bw: initial_state_bw,
                                                          self.keep_prob: 1.0})
        confusion_matrix = tf.confusion_matrix(tf.argmax(test_labels, axis=1),
                                               tf.argmax(logits, axis=1)).eval(session=sess)

        return loss, accuracy, confusion_matrix

    def predict(self, sess, pred_input):
        # create initial zero state
        initial_state_fw = np.zeros((self.n_layers, len(pred_input), self.n_hidden))
        initial_state_bw = np.zeros((self.n_layers, len(pred_input), self.n_hidden))
        logits = sess.run(self.logits, feed_dict={self.input: pred_input,
                                                  self.initial_state_fw: initial_state_fw,
                                                  self.initial_state_bw: initial_state_bw,
                                                  self.keep_prob: 1.0})
        return tf.nn.softmax(logits).eval(session=tf.Session())

