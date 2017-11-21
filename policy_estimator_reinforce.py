import tensorflow as tf


class PolicyEstimatorReinforce(object):
    def __init__(self, name):
        self._build_model(name)

    def _build_model(self, name):
        with tf.name_scope(name):
            self.state = tf.placeholder(tf.float32, [1, 15, 15, 3], name="state")
            self.action = tf.placeholder(tf.int32, [1], name="action")
            self.advantage = tf.placeholder(tf.float32, [1], name="advantage")

            with tf.name_scope("predictions"):
                self.conv1 = tf.layers.conv2d(self.state, filters=64, kernel_size=[7, 7], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv2 = tf.layers.conv2d(self.conv1, filters=64, kernel_size=[5, 5], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv3 = tf.layers.conv2d(self.conv2, filters=64, kernel_size=[3, 3], strides=(1, 1), padding="same", activation=tf.nn.relu)
                self.conv3_flat = tf.reshape(self.conv3, shape=[-1, 15 * 15 * 64])
                self.dense = tf.layers.dense(self.conv3_flat, units=512, activation=tf.nn.relu)
                self.action_probs = tf.layers.dense(self.dense, units=15*15, activation=tf.nn.softmax)

            with tf.name_scope("loss"):
                self.picked_action_prob = tf.gather(self.action_probs[0], self.action)
                self.loss = -tf.log(self.picked_action_prob) * self.advantage

            with tf.name_scope("train"):
                # self.train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)

    def predict(self, session, state):
        return session.run(self.action_probs, feed_dict={self.state: [state]})[0]

    def update(self, session, state, advantage, action):
        feed_dict = {self.state: [state], self.advantage: [advantage], self.action: [action]}
        session.run(self.train_op, feed_dict)
