import tensorflow as tf


class ValueEstimator(object):
    def __init__(self, name, discount_factor):
        self.discount_factor = discount_factor
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.state = tf.placeholder(tf.float32, [None, 15, 15, 3], name="state")
            self.target_value = tf.placeholder(tf.float32, [None, 1], name="target_value")
            self.dropout_keep_prob = tf.placeholder(tf.float32)

            with tf.name_scope("value"):
                self.conv1 = tf.layers.conv2d(self.state, filters=64, kernel_size=[5, 5], strides=(1, 1), padding="valid", activation=tf.nn.relu)
                self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                # self.conv4 = tf.layers.conv2d(self.conv3, filters=32, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                # self.conv5 = tf.layers.conv2d(self.conv4, filters=32, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv3_flat = tf.reshape(self.conv3, shape=[-1, 11 * 11 * 64])
                # self.conv5_flat = tf.reshape(self.conv5, shape=[-1, 11 * 11 * 32])
                self.dense = tf.nn.dropout(tf.layers.dense(self.conv3_flat, units=512, activation=tf.nn.relu), self.dropout_keep_prob)
                # self.dense = tf.layers.dense(self.conv5_flat, units=512, activation=tf.nn.relu)
                self.value = tf.layers.dense(self.dense, units=1)

            with tf.name_scope("loss"):
                self.loss = tf.squared_difference(self.value, self.target_value)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(self.loss)
                # self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def td_error(self, session, state, reward, next_state):
        value_for_state = session.run(self.value, feed_dict={self.state: [state], self.dropout_keep_prob: 1.0})[0][0]
        value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state], self.dropout_keep_prob: 1.0})[0][0]
        td_target = reward + self.discount_factor * value_for_next_state
        td_error = td_target - value_for_state

        return td_error

    def update(self, session, state, reward, next_state, done):
        target_value = reward
        if not done:
            value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state], self.dropout_keep_prob: 1.0})[0][0]
            target_value += self.discount_factor * value_for_next_state

        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[target_value]], self.dropout_keep_prob: 0.75})

    def update_done(self, session, state, reward):
        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[reward]], self.dropout_keep_prob: 0.75})
