from base import BaseModel
from configs import args
from model.model_utils import create_multi_rnn_cell
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

class Decoder(BaseModel):
    def __init__(self, vocab_size, word_embeddings):
        super().__init__('decoder')
        self.vocab_size = vocab_size
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):

            self.embedding = tf.constant(word_embeddings, dtype=tf.float32)
            
            self.decoder_cell = create_multi_rnn_cell(args['rnn_type'],
                                                      args['rnn_size'],
                                                      args['keep_prob'],
                                                      args['num_layer'])
            # Hidden state that gets feed to the decoder cell, this is the hidden state of the context concatenated with the latent
            self.state_dense = tf.compat.v1.layers.Dense(units=args['rnn_size'],    
                                                         activation=tf.nn.relu,     
                                                         kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                             0.0, 0.01),    
                                                         bias_initializer=tf.compat.v1.zeros_initializer,
                                                         name='decoder/state_dense')
            # Output decoded from the decoder cell will be plugged into this layer
            self.output_dense = tf.compat.v1.layers.Dense(units=self.vocab_size,
                                                          kernel_initializer=tf.compat.v1.truncated_normal_initializer(
                                                              0.0, 0.01),
                                                          bias_initializer=tf.compat.v1.zeros_initializer,
                                                          name='decoder/output_dense')

    def __call__(self, context_with_latent, is_training=False, decoder_inputs=None):
        if is_training and (decoder_inputs is None):
            raise ValueError(" # decoder_inputs are required if is_traning is True")
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            init_state_tuple = ()
            for i in range(args['num_layer']):
                init_state = self.state_dense(context_with_latent[i])
                init_state_tuple = init_state_tuple + (init_state, )
            if is_training:  # training
                embedded_inputs = tf.nn.embedding_lookup(params=self.embedding, ids=decoder_inputs)
                decoder_lengths = tf.math.count_nonzero(decoder_inputs, -1, dtype=tf.int32)
                samp = tfa.seq2seq.sampler.TrainingSampler()
                train_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=samp,
                                                         output_layer=self.output_dense)
                train_output, _, _ = tfa.seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    swap_memory=True,
                    maximum_iterations=tf.reduce_max(input_tensor=decoder_lengths),
                    decoder_init_input=embedded_inputs,
                    decoder_init_kwargs={
                        'initial_state': init_state_tuple, 'sequence_length': decoder_lengths
                    })
                logits = train_output.rnn_output
                sample_id = train_output.sample_id
                return logits, sample_id
            else:  # inferring
                samp = tfa.seq2seq.sampler.SampleEmbeddingSampler()
                infer_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=samp,
                                                         output_layer=self.output_dense)
                infer_output, _, _ = tfa.seq2seq.dynamic_decode(
                    decoder=infer_decoder,
                    swap_memory=True,
                    maximum_iterations=args['max_len'] + 1,
                    decoder_init_input=self.embedding,
                    decoder_init_kwargs={
                        'start_tokens': tf.tile(tf.constant([args['SOS_ID']], dtype=tf.int32), [args['batch_size']]),
                        'end_token': args['EOS_ID'],
                        'initial_state': init_state_tuple # ucomment this when not use beam search:  tfa.seq2seq.tile_batch(init_state_tuple, args['beam_width'])
                    })
                # infer_predicted_ids = infer_output.predicted_ids[:, :, 0]  # select the first sentence
                infer_predicted_ids = infer_output.sample_id
                # The return is a list consisting of only one element whose diamention is (batch_size, max_len)
                return infer_predicted_ids