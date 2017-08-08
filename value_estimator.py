import tensorflow as tf


class ValueEstimator(object):
    def __init__(self, name, discount_factor):
        self.discount_factor = discount_factor
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.state = tf.placeholder(tf.float32, [1, 15, 15, 3], name="state")
            self.target_value = tf.placeholder(tf.float32, [1, 1], name="target_value")

            with tf.name_scope("value"):
                self.conv_W1 = tf.Variable(tf.truncated_normal([5, 5, 3, 128], stddev=0.1))
                self.conv_b1 = tf.Variable(tf.zeros([128]))
                self.conv1 = tf.nn.relu(tf.nn.conv2d(self.state, self.conv_W1, strides=[1, 1, 1, 1], padding='VALID') + self.conv_b1)
                # self.conv1 = tf.layers.conv2d(self.state, filters=128, kernel_size=[5, 5], strides=(1, 1), padding="valid", activation=tf.nn.relu)
                self.conv_W2 = tf.Variable(tf.truncated_normal([4, 4, 128, 64], stddev=0.1))
                self.conv_b2 = tf.Variable(tf.zeros([64]))
                self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.conv_W2, strides=[1, 1, 1, 1], padding='SAME') + self.conv_b2)
                # self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[4, 4], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv_W3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
                self.conv_b3 = tf.Variable(tf.zeros([64]))
                self.conv3 = tf.nn.relu(tf.nn.conv2d(self.conv2, self.conv_W3, strides=[1, 1, 1, 1], padding='SAME') + self.conv_b3)
                # self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv_W4 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
                self.conv_b4 = tf.Variable(tf.zeros([64]))
                self.conv4 = tf.nn.relu(tf.nn.conv2d(self.conv3, self.conv_W4, strides=[1, 1, 1, 1], padding='SAME') + self.conv_b4)
                # self.conv4 = tf.layers.conv2d(self.conv3, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv_W5 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
                self.conv_b5 = tf.Variable(tf.zeros([64]))
                self.conv5 = tf.nn.relu(tf.nn.conv2d(self.conv4, self.conv_W5, strides=[1, 1, 1, 1], padding='SAME') + self.conv_b5)
                # self.conv5 = tf.layers.conv2d(self.conv4, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                # self.conv3_flat = tf.reshape(self.conv3, shape=[-1, 11 * 11 * 64])
                self.conv5_flat = tf.reshape(self.conv5, shape=[-1, 11 * 11 * 64])
                # self.dense = tf.layers.dense(self.conv3_flat, units=512, activation=tf.nn.relu)
                self.dense = tf.layers.dense(self.conv5_flat, units=512, activation=tf.nn.relu)
                self.value = tf.layers.dense(self.dense, units=1)

                tf.summary.histogram("conv_W1", self.conv_W1)
                tf.summary.histogram("conv_b1", self.conv_b1)
                tf.summary.histogram("conv_W2", self.conv_W2)
                tf.summary.histogram("conv_b2", self.conv_b2)
                tf.summary.histogram("conv_W3", self.conv_W3)
                tf.summary.histogram("conv_b3", self.conv_b3)
                tf.summary.histogram("conv_W4", self.conv_W4)
                tf.summary.histogram("conv_b4", self.conv_b4)
                tf.summary.histogram("conv_W5", self.conv_W5)
                tf.summary.histogram("conv_b5", self.conv_b5)

            with tf.name_scope("loss"):
                self.loss = tf.squared_difference(self.value, self.target_value)

                # tf.summary.scalar("loss", self.loss)

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
                # self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def td_error(self, session, state, reward, next_state):
        value_for_state = session.run(self.value, feed_dict={self.state: [state]})[0][0]
        value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state]})[0][0]
        td_target = reward + self.discount_factor * value_for_next_state
        td_error = td_target - value_for_state

        return td_error

    def update(self, session, state, reward, next_state, done):
        target_value = reward
        if not done:
            value_for_next_state = session.run(self.value, feed_dict={self.state: [next_state]})[0][0]
            target_value += self.discount_factor * value_for_next_state

        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[target_value]]})

    def update_done(self, session, state, reward):
        session.run(self.train_op, feed_dict={self.state: [state], self.target_value: [[reward]]})
