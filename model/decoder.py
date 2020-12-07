from base import BaseModel
from configs import args
from model.model_utils import create_multi_rnn_cell
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa


class Decoder(BaseModel):
    def __init__(self, vocab_size):
        super().__init__('decoder')
        self.vocab_size = vocab_size
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            self.embedding = tf.compat.v1.get_variable('lookup_table', [vocab_size, args['embed_dims']])
            self.decoder_cell = create_multi_rnn_cell(args['rnn_type'],
                                                      args['rnn_size'],
                                                      args['keep_prob'],
                                                      args['num_layer'])
            # Hidden state that gets feed to the decoder cell
            self.state_dense = tf.compat.v1.layers.Dense(units=args['rnn_size'],    # shape of output space
                                                         activation=tf.nn.relu,     # activation function
                                                         kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                             0.0, 0.01),    # initialize weights from a truncated normal
                                                         # distribution (same to Gaussian but discarding values more than 2 std from the mean)
                                                         bias_initializer=tf.compat.v1.zeros_initializer,
                                                         name='decoder/state_dense')
            # Output decoded from the decoder cell will be plugged into this layer
            self.output_dense = tf.compat.v1.layers.Dense(units=self.vocab_size,
                                                          kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                              0.0, 0.01),
                                                          bias_initializer=tf.compat.v1.zeros_initializer,
                                                          name='decoder/output_dense')

    def __call__(self, context_with_latent, is_training=False, decoder_inputs=None):
        # diamention of context_with_latent:
        # (num_layer, (batch_size, state_dim+latent_size))
        if is_training and (decoder_inputs is None):
            raise ValueError(" # decoder_inputs are required if is_traning is True")
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            init_state_tuple = ()
            for i in range(args['num_layer']):
                init_state = self.state_dense(context_with_latent[i])
                init_state_tuple = init_state_tuple + (init_state,)
            if is_training:  # training
                embedded_inputs = tf.nn.embedding_lookup(params=self.embedding, ids=decoder_inputs)
                decoder_lengths = tf.math.count_nonzero(decoder_inputs, -1, dtype=tf.int32)
                trainingHelper = tfa.seq2seq.TrainingSampler()
                train_helper = trainingHelper.initialize(inputs=embedded_inputs,
                                                         sequence_length=decoder_lengths)
                samp = tfa.seq2seq.sampler.TrainingSampler()
                # train_decoder =tfa.seq2seq.BasicDecoder(cell=self.decoder_cell,
                #                                                helper=train_helper, sampler=samp
                #                                                initial_state=init_state_tuple,
                #                                                output_layer=self.output_dense)
                train_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=samp,
                                                         output_layer=self.output_dense)
                train_output, _, _ = tfa.seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    swap_memory=True,
                    maximum_iterations=tf.reduce_max(input_tensor=decoder_lengths),
                    decoder_init_input=embedded_inputs,
                    decoder_init_kwargs={
                        'initial_state': init_state_tuple
                    })
                logits = train_output.rnn_output  # (batch_size, dec_max_lentgh, vocab_size) 概率分布
                sample_id = train_output.sample_id  # (batch_size, dec_max_length) 解码结果
                return logits, sample_id
            else:  # inferring
                infer_decoder = tfa.seq2seq.BeamSearchDecoder(
                    cell=self.decoder_cell,
                    beam_width=args['beam_width'],
                    output_layer=self.output_dense)
                infer_output, _, _ = tfa.seq2seq.dynamic_decode(
                    decoder=infer_decoder,
                    swap_memory=True,
                    maximum_iterations=args['max_len'] + 1,
                    decoder_init_input=self.embedding,
                    decoder_init_kwargs={
                        'start_tokens': tf.tile(tf.constant([args['SOS_ID']], dtype=tf.int32), [args['batch_size']]),
                        'end_token': args['EOS_ID'],
                        'initial_state': tfa.seq2seq.tile_batch(init_state_tuple, args['beam_width'])
                    })
                infer_predicted_ids = infer_output.predicted_ids[:, :, 0]  # select the first sentence

                # The return is a list consisting of only one element whose diamention is (batch_size, max_len)
                return infer_predicted_ids
