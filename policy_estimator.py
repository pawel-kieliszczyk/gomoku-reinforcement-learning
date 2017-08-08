import tensorflow as tf


class PolicyEstimator(object):
    def __init__(self, name):
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.state = tf.placeholder(tf.float32, [1, 15, 15, 3], name="state")
            self.action = tf.placeholder(tf.int32, [1], name="action")
            self.advantage = tf.placeholder(tf.float32, [1], name="advantage")

            with tf.name_scope("predictions"):
                self.conv_W1 = tf.Variable(tf.truncated_normal([7, 7, 3, 128], stddev=0.1))
                self.conv_b1 = tf.Variable(tf.zeros([128]))
                self.conv1 = tf.nn.relu(tf.nn.conv2d(self.state, self.conv_W1, strides=[1, 1, 1, 1], padding='VALID') + self.conv_b1)
                # self.conv1 = tf.layers.conv2d(self.state, filters=128, kernel_size=[7, 7], strides=(1, 1), padding="valid", activation=tf.nn.relu)
                self.conv_W2 = tf.Variable(tf.truncated_normal([5, 5, 128, 64], stddev=0.1))
                self.conv_b2 = tf.Variable(tf.zeros([64]))
                self.conv2 = tf.nn.relu(tf.nn.conv2d(self.conv1, self.conv_W2, strides=[1, 1, 1, 1], padding='SAME') + self.conv_b2)
                # self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=tf.nn.relu)
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
                # self.conv3_flat = tf.reshape(self.conv3, shape=[-1, 9 * 9 * 64])
                self.conv5_flat = tf.reshape(self.conv5, shape=[-1, 9 * 9 * 64])
                # self.dense = tf.layers.dense(self.conv3_flat, units=512, activation=tf.nn.relu)
                self.dense = tf.layers.dense(self.conv5_flat, units=512, activation=tf.nn.relu)
                self.action_probs = tf.layers.dense(self.dense, units=15*15, activation=tf.nn.softmax)

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
                self.picked_action_prob = tf.gather(self.action_probs[0], self.action)
                self.loss = -tf.log(self.picked_action_prob) * self.advantage

            with tf.name_scope("train"):
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.0000001).minimize(self.loss)
                # self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)

    def predict(self, session, state):
        return session.run(self.action_probs, feed_dict={self.state: [state]})[0]

    def update(self, session, state, advantage, action):
        feed_dict = {self.state: [state], self.advantage: [advantage], self.action: [action]}
        session.run(self.train_op, feed_dict)
