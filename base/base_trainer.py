import tensorflow as tf


class BaseTrainer(object):
    def __init__(self, sess, model):
        self.sess = sess
        self.model = model
        self.init = tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        raise NotImplementedError