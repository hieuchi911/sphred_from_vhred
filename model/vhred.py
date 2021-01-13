from base import BaseModel
from configs import args
from .encoder import Encoder
from .decoder import Decoder
from .model_utils import reparamter_trick, kl_weights_fn, kl_loss_fn
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa
import os


class VHRED(BaseModel):
    def __init__(self, dataLoader, build_graph=True, is_train=True):
        super().__init__('VHRED')
        self.dataLoader = dataLoader
        # Initialize 3 modules: Encoder RNN, Context RNN and Decoder RNN
        self.encoder_RNN = Encoder(dataLoader.vocab_size, word_embeddings=dataLoader.embedding_matrix)
        self.context_RNN = Encoder(dataLoader.vocab_size, is_embedding=False)
        self.decoder_RNN = Decoder(dataLoader.vocab_size, word_embeddings=dataLoader.embedding_matrix)
        # print("\n\n\n\n\n\n\n\n\n--------------------------\n\nErMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n\n--------------------------\n\n\n\n\n\n\n\n\n")
        if build_graph:
            self.build_global_helper()
            self.build_encoder_graph()
            self.build_context_graph()
            self.build_prior_graph()
            if is_train:
                self.build_encoder_current_step_graph()
                self.build_posterior_graph()
                self.build_train_decoder_graph()
                self.build_backward_graph()
            self.init_saver()
            self.build_infer_decoder_graph()

    def init_saver(self):
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)
        p = os.path.dirname(args['vhred_ckpt_dir'])
        if not os.path.exists(p):
            os.makedirs(p)

    def build_global_helper(self):
        self.encoder_inputs = tf.compat.v1.placeholder(tf.int32,
                                                       [None, None, None])  # (batch_size, utterance_num, max_len)
        self.decoder_inputs = tf.compat.v1.placeholder(tf.int32, [None, None])  # (batch_size, max_len)
        self.decoder_targets = tf.compat.v1.placeholder(tf.int32, [None, None])  # (batch_size, max_len)
        self.encoder_lengths = tf.math.count_nonzero(self.encoder_inputs, -1, dtype=tf.int32)  # (batch_size) 每行句子的长度
        self.decoder_lengths = tf.math.count_nonzero(self.decoder_inputs, -1, dtype=tf.int32)

        self.global_step = tf.Variable(0, trainable=False)
        self.dec_max_len = tf.reduce_max(input_tensor=self.decoder_lengths, name="dec_max_len")
        self.decoder_weights = tf.sequence_mask(self.decoder_lengths, self.dec_max_len, dtype=tf.float32)

        self.lr = tf.Variable(args['learning_rate'], trainable=False)
        self.new_lr = tf.compat.v1.placeholder(tf.float32, [])
        self.update_lr_op = tf.compat.v1.assign(self.lr, self.new_lr)

    def build_encoder_graph(self):
        with tf.compat.v1.variable_scope('encoder', reuse=tf.compat.v1.AUTO_REUSE):
            self.enc_state_list = []  # (utterance_num, batch_size, state_dim)
            for i in range(args['num_pre_utterance']):  # 3 utterances said before
                current_utterance_inputs = self.encoder_inputs[:, i, :]  # (batch_size, max_len), the i'th speaking turn of all dialogs in the batch
                outputs, states = self.encoder_RNN(current_utterance_inputs) # all outputs and states for the i'th utterances, each state here captures corresponding sequence said
                self.enc_state_list.append(states[-1])  # states[-1] since encoderRNN is a 2-layer RNN network, just need the later layer

    def build_encoder_current_step_graph(self):
        with tf.compat.v1.variable_scope('encoder/current_step', reuse=tf.compat.v1.AUTO_REUSE):
            outputs, states = self.encoder_RNN(self.decoder_inputs) # The fourth 4th utterance as response to the third
            self.current_step_state = states  # (num_layer, (batch_size, state_dim)), so not assign states[-1] to current_step_states

    def build_context_graph(self):
        with tf.compat.v1.variable_scope('context', reuse=tf.compat.v1.AUTO_REUSE):
            self.enc_state_list = tf.transpose(a=self.enc_state_list,
                                               perm=[1, 0, 2])  # (batch_size, utterance_num, state_dim)
            outputs, states = self.context_RNN(self.enc_state_list)
            self.context_state = states  # (num_layer, (batch_size, state_dim)), so not assign states[-1] to context_state

    def build_prior_graph(self):
        with tf.compat.v1.variable_scope('prior', reuse=tf.compat.v1.AUTO_REUSE):
            # 2 layers neural network: prior_dense_1 and prior_dense_2
            self.prior_dense_1 = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                           activation=tf.nn.tanh,
                                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                               0.0, 0.01),
                                                           bias_initializer=tf.compat.v1.zeros_initializer,
                                                           name='prior_dense_1')
            self.prior_dense_2 = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                           activation=tf.nn.tanh,
                                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                               0.0, 0.01),
                                                           bias_initializer=tf.compat.v1.zeros_initializer,
                                                           name='prior_dense_2')
            self.prior_mean = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                        kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                            0.0, 0.01),
                                                        bias_initializer=tf.compat.v1.zeros_initializer,
                                                        name='prior_mean')
            self.prior_log_var = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                           kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                               0.0, 0.01),
                                                           bias_initializer=tf.compat.v1.zeros_initializer,
                                                           name='prior_log_var', activation='softplus')
            self.prior_z_tuple = ()  # (num_layer, (batch_size, latent_size))
            for i in range(args['num_layer']):
                prior_dense_1_out = self.prior_dense_1(self.context_state[i])
                prior_dense_2_out = self.prior_dense_2(prior_dense_1_out)
                self.prior_mean_value = self.prior_mean(prior_dense_2_out)
                self.prior_log_var_value = self.prior_log_var(prior_dense_2_out)
                prior_z = reparamter_trick(self.prior_mean_value, self.prior_log_var_value)  # (batch_size, latent_size)
                self.prior_z_tuple = self.prior_z_tuple + (prior_z,)

    def build_posterior_graph(self):
        with tf.compat.v1.variable_scope('posterior', reuse=tf.compat.v1.AUTO_REUSE):
            # (num_layer,(batch_size, 2*state_dim)), num_layer as first dim since each context_state and current_step_state are from 2-layer rnns
            self.context_with_current_step = tf.concat([self.context_state, self.current_step_state], -1)
            # 2 layers neural network
            self.posterior_dense_1 = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                               activation=tf.nn.tanh,
                                                               kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                                   0.0, 0.01),
                                                               bias_initializer=tf.compat.v1.zeros_initializer,
                                                               name='posterior_dense_1')
            self.posterior_mean = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                            kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                                0.0, 0.01),
                                                            bias_initializer=tf.compat.v1.zeros_initializer,
                                                            name='posterior_mean')
            self.posterior_log_var = tf.compat.v1.layers.Dense(units=args['latent_size'],
                                                               kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                                   0.0, 0.01),
                                                               bias_initializer=tf.compat.v1.zeros_initializer,
                                                               name='posterior_log_var', activation='softplus')
            self.posterior_z_tuple = ()  # (num_layer, (batch_size, latent_size))
            for i in range(args['num_layer']):
                posterior_dense_1_out = self.posterior_dense_1(
                    self.context_with_current_step[i])  # (batch_size, latent_dim)
                self.posterior_mean_value = self.posterior_mean(posterior_dense_1_out)
                self.posterior_log_var_value = self.posterior_log_var(posterior_dense_1_out)
                posterior_z = reparamter_trick(self.posterior_mean_value, self.posterior_log_var_value)
                self.posterior_z_tuple = self.posterior_z_tuple + (posterior_z,)

    def build_train_decoder_graph(self):
        with tf.compat.v1.variable_scope('train/decoder', reuse=tf.compat.v1.AUTO_REUSE):
            # (num_layer, (batch_size, state_dim+latent_size))
            self.context_with_latent_train = tf.concat([self.context_state, self.posterior_z_tuple], -1)
            self.train_logits, self.train_sample_id, self.embeddings = self.decoder_RNN(
                context_with_latent=self.context_with_latent_train,
                is_training=True,
                decoder_inputs=self.decoder_inputs)

    def build_backward_graph(self):
        self.nll_loss = tf.reduce_mean(input_tensor=tfa.seq2seq.sequence_loss(logits=self.train_logits,
                                                                              targets=self.decoder_targets,
                                                                              weights=self.decoder_weights,
                                                                              average_across_timesteps=False,
                                                                              average_across_batch=True))
        self.kl_weights = kl_weights_fn(self.global_step)
        self.kl_loss = kl_loss_fn(mean_1=self.posterior_mean_value,
                                  log_var_1=self.posterior_log_var_value,
                                  mean_2=self.prior_mean_value,
                                  log_var_2=self.prior_log_var_value)
        self.loss = self.nll_loss + self.kl_weights * self.kl_loss
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr)
        self.tvars = tf.compat.v1.trainable_variables()  # trainable_variables
        grads = tf.gradients(ys=self.loss, xs=self.tvars)
        clip_grads, _ = tf.clip_by_global_norm(grads, args['clip_norm'])
        self.train_op = optimizer.apply_gradients(zip(clip_grads, self.tvars), global_step=self.global_step)

    def build_infer_decoder_graph(self):
        with tf.compat.v1.variable_scope('infer/decoder', reuse=tf.compat.v1.AUTO_REUSE):
            self.context_with_latent_infer = tf.concat([self.context_state, self.prior_z_tuple], -1, name='context_with_latent_infer')
            self.infer_decoder_ids = self.decoder_RNN(context_with_latent=self.context_with_latent_infer,
                                                      is_training=False)
            self.infer_decoder_logits = tf.one_hot(self.infer_decoder_ids, depth=self.dataLoader.vocab_size)
            self.nll_loss_test = tf.reduce_mean(input_tensor=tfa.seq2seq.sequence_loss(logits=self.infer_decoder_logits,
                                                                                       targets=self.decoder_targets,
                                                                                       weights=self.decoder_weights,
                                                                                       average_across_timesteps=False,
                                                                                       average_across_batch=True))
            self.loss_test = self.nll_loss_test + self.kl_weights * self.kl_loss

    def encoder_state_session(self, sess, enc_inp):
        result = sess.run(self.enc_state_list, feed_dict={self.encoder_inputs: enc_inp})
        # result = sess.run(self.current_utterance_inputs, feed_dict={self.encoder_inputs:enc_inp})
        return result, state

    def encoder_current_step_session(self, sess, dec_inp):
        result = sess.run(self.current_step_state, feed_dict={self.decoder_inputs: dec_inp})
        return result

    def context_state_session(self, sess, enc_inp):
        result = sess.run(self.context_state, feed_dict={self.encoder_inputs: enc_inp})
        return result

    def prior_z_session(self, sess, enc_inp):
        result = sess.run(self.prior_z_tuple, feed_dict={self.encoder_inputs: enc_inp})
        return result

    def posterior_z_session(self, sess, enc_inp, dec_inp):
        result = sess.run(self.posterior_z_tuple,
                          feed_dict={self.encoder_inputs: enc_inp, self.decoder_inputs: dec_inp})
        return result

    def train_decoder_session(self, sess, enc_inp, dec_inp):
        train_logits, train_sample_id = sess.run([self.train_logits, self.train_sample_id],
                                                 feed_dict={self.encoder_inputs: enc_inp, self.decoder_inputs: dec_inp})
<<<<<<< HEAD
        print("\n\n\n*******************************\tOutput dense logits: ", train_logits[0][0], "\n*******************************\n\n\n")
        array_of = {}
        array_gt_3 = {}
        tracker = 0
        for i in train_logits[0][0]:
          if i > 0:
            array_of[tracker] = i
          if i >= train_logits[0][0][3]:
            array_gt_3[tracker] = i
          tracker += 1
        print("len of train_logits[0]: ", len(train_logits[0]), "and type is: ", type(train_logits[0][0]))
        print("array_of: ", array_of, "\nAnd length of array_of is: ", len(array_of))
        print("array_gt_3: ", array_gt_3, "\nAnd length of array_gt_3 is: ", len(array_gt_3))
        print("\n\n\n*******************************\tOutput dense ids: ", train_sample_id, "\n*******************************\n\n\n")
=======
        print("\n\n\n*******************************\tOutput dense logits: ", train_logits, "\n************************************\n\n\n")
        print("\n\n\n*******************************\tOutput dense ids: ", train_sample_id, "\n************************************\n\n\n")
>>>>>>> 9e1285de6039b36ba44fc97a89df5849858aad04
        return train_logits, train_sample_id

    def infer_decoder_session(self, sess, enc_inp):
        infer_decoder_ids = sess.run([self.infer_decoder_ids],
                                     feed_dict={self.encoder_inputs: enc_inp})
        print("\n\n\n*******************************latent plus context output:\n", sess.run(self.context_with_latent_infer, feed_dict={self.encoder_inputs: enc_inp}))
<<<<<<< HEAD
=======
        # print("Output dense: ", sess.run(self.cell_outputs(input_shape=(args['batch_size'], args['rnn_size'])).output, feed_dict={self.encoder_inputs: enc_inp}), "\n************************************\n\n\n")
>>>>>>> 9e1285de6039b36ba44fc97a89df5849858aad04
        return infer_decoder_ids

    def kl_loss_session(self, sess, enc_inp, dec_inp, dec_tar):
        kl_loss = sess.run(self.kl_loss, feed_dict={self.encoder_inputs: enc_inp,
                                                    self.decoder_inputs: dec_inp,
                                                    self.decoder_targets: dec_tar})
        return kl_loss

    def loss_session(self, sess, enc_inp, dec_inp, dec_tar):
        loss = sess.run(self.loss, feed_dict={self.encoder_inputs: enc_inp,
                                              self.decoder_inputs: dec_inp,
                                              self.decoder_targets: dec_tar})
        return loss

    def train_session(self, sess, enc_inp, dec_inp, dec_tar):
        fetches = [self.train_op, self.loss, self.nll_loss, self.kl_loss, self.global_step]
        feed_dict = {self.encoder_inputs: enc_inp,
                     self.decoder_inputs: dec_inp,
                     self.decoder_targets: dec_tar}
        _, loss, nll_loss, kl_loss, global_step = sess.run(fetches, feed_dict)
        return {'loss': loss, 'nll_loss': nll_loss, 'kl_loss': kl_loss, 'global_step': global_step}

    def test_session(self, sess, enc_inp, dec_inp, dec_tar):
        # fetches = [self.loss_test, self.nll_loss_test, self.kl_loss]
        fetches = [self.loss, self.nll_loss, self.kl_loss]
        feed_dict = {self.encoder_inputs: enc_inp,
                     self.decoder_inputs: dec_inp,
                     self.decoder_targets: dec_tar}
        loss_test, nll_loss_test, kl_loss = sess.run(fetches, feed_dict)
        return {'loss_test': loss_test, 'nll_loss_test': nll_loss_test, 'kl_loss': kl_loss}
