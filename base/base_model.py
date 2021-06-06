import tensorflow as tf
import os


class BaseModel(object):
    def __init__(self, _scope):
        self._scope = _scope
        self.ops = {}

    def save(self, saver, sess, ckpt_dir):
        print("Saving model...")
        saver.save(sess, ckpt_dir, global_step=2)
        print("Model saved")

    def load(self, saver, sess, ckpt_dir):
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_dir))
        saver.restore(sess, latest_ckpt)
        print("Model loaded")
