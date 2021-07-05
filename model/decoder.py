from base import BaseModel
from configs import args
from model.model_utils import create_multi_rnn_cell
import numpy as np

import tensorflow as tf
import tensorflow_addons as tfa

class TrainingSamp(tfa.seq2seq.sampler.TrainingSampler):
    def sample(self, time, outputs, state):
        # del state
        # sample_ids = tf.cast(tf.argmax(outputs, axis=-1), tf.int32)
        # return sample_ids
        logits = []
        top_p = 0.05
        if top_p > 0.0:
            # 1 - sort the tensor elements and their indices
            sorted_tensor = tf.sort(outputs, axis=-1, direction='DESCENDING')
            sorted_indices = tf.argsort(outputs, axis=-1, direction='DESCENDING')

            # 2 - convert to probability by using softmax
            proba_tensor = tf.nn.softmax(sorted_tensor, axis=-1)
            accumulated_proba = tf.cumsum(proba_tensor, axis=-1)  # elements in this tensor increases

            true_ = tf.tile([[True for i in range(20)]], [tf.shape(accumulated_proba)[0], 1])

            # A list of coordinates to update. [[0,a], [0, b], ..., [3, h]...]:
            keep_indices = m = tf.concat([tf.where(accumulated_proba < top_p), tf.where(true_)], axis=0)
            # Turn [[0,a], [0, b], ..., [3, h]...] to [[0,0], [0, 0], ..., [3, 0]...]:
            keeping_indices = tf.linalg.matmul(keep_indices, tf.constant([[1, 0], [0, 0]], dtype=tf.int64))

            logit_indices = tf.gather_nd(sorted_indices,
                                         keep_indices)  # ---> [a', b', ..., h'], BUT supposed to be [[0,a'], [0, b'], ..., [3, h']...]
            diagonal = tf.linalg.diag(logit_indices)  # create diagonal matrix from [a', b', ..., h']
            zero_one = tf.tile(tf.constant([[0, 1]], dtype=tf.int64),
                               [tf.shape(keep_indices)[0], 1])  # create [[0, 1]..., [0, 1]]
            logit_indices = tf.linalg.matmul(tf.cast(diagonal, dtype=tf.int64),
                                             zero_one)  # multiply diagonal (6, 6) with 0,1 (6, 2) to generate THE NEEDED [[0,a'], [0, b'], ..., [3, h']...] ABOVE

            logit_indices = logit_indices + keeping_indices
            # The shape of the corresponding dense tensor, same as `c`.
            # create an array of the number of elements need replacing
            values = tf.fill([tf.shape(keep_indices)[0]], 2000000.0)

            # create this shape, the same to the original logits shape
            shape = tf.shape(outputs, out_type=tf.int64)

            # this return a scattered matrix of shape shape, elements at indices specified
            # in remove_indices will be filled with the corresponding value in values
            delta = tf.scatter_nd(logit_indices, values, shape)

            # combine delta and outputs, logits needs keeping stay the same while other changes to -inf
            logits = outputs + delta
            sample_ids = categorical_sample(logits=logits, seed=50)
        return sample_ids


class NucleusSampler(tfa.seq2seq.sampler.SampleEmbeddingSampler):
    def sample(self, time, outputs, state):
        logits = []
        top_p = 0.05
        if top_p > 0.0:
            # 1 - sort the tensor elements and their indices
            sorted_tensor = tf.sort(outputs, axis=-1, direction='DESCENDING')
            sorted_indices = tf.argsort(outputs, axis=-1, direction='DESCENDING')

            # 2 - convert to probability by using softmax
            proba_tensor = tf.nn.softmax(sorted_tensor, axis=-1)
            accumulated_proba = tf.cumsum(proba_tensor, axis=-1)  # elements in this tensor increases
            
            true_ = tf.tile([[True for i in range(20)]], [tf.shape(accumulated_proba)[0], 1])

            # A list of coordinates to update. [[0,a], [0, b], ..., [3, h]...]:
            keep_indices = m = tf.concat([tf.where(accumulated_proba < top_p), tf.where(true_)], axis=0)
            # Turn [[0,a], [0, b], ..., [3, h]...] to [[0,0], [0, 0], ..., [3, 0]...]:
            keeping_indices = tf.linalg.matmul(keep_indices, tf.constant([[1, 0], [0, 0]], dtype=tf.int64))

            logit_indices = tf.gather_nd(sorted_indices,
                                         keep_indices)  # ---> [a', b', ..., h'], BUT supposed to be [[0,a'], [0, b'], ..., [3, h']...]
            diagonal = tf.linalg.diag(logit_indices)  # create diagonal matrix from [a', b', ..., h']
            zero_one = tf.tile(tf.constant([[0, 1]], dtype=tf.int64),
                               [tf.shape(keep_indices)[0], 1])  # create [[0, 1]..., [0, 1]]
            logit_indices = tf.linalg.matmul(tf.cast(diagonal, dtype=tf.int64),
                                             zero_one)  # multiply diagonal (6, 6) with 0,1 (6, 2) to generate THE NEEDED [[0,a'], [0, b'], ..., [3, h']...] ABOVE

            logit_indices = logit_indices + keeping_indices
            # The shape of the corresponding dense tensor, same as `c`.
            # create an array of the number of elements need replacing
            values = tf.fill([tf.shape(keep_indices)[0]], 2000000.0)

            # create this shape, the same to the original logits shape
            shape = tf.shape(outputs, out_type=tf.int64)

            # this return a scattered matrix of shape shape, elements at indices specified
            # in remove_indices will be filled with the corresponding value in values
            delta = tf.scatter_nd(logit_indices, values, shape)

            # combine delta and outputs, logits needs keeping stay the same while other changes to -inf
            logits = outputs + delta
            # sample_ids = categorical_sample(logits=logits)
        return categorical_sample(logits=logits, seed=50)

# The following sample functions (_call_sampler, bernoulli_sample,
# categorical_sample) mimic TensorFlow Probability distribution semantics.
def _call_sampler(sample_n_fn, sample_shape, name=None):
    """Reshapes vector of samples."""
    with tf.name_scope(name or "call_sampler"):
        sample_shape = tf.convert_to_tensor(
            sample_shape, dtype=tf.int32, name="sample_shape"
        )
        # Ensure sample_shape is a vector (vs just a scalar).
        pad = tf.cast(tf.equal(tf.rank(sample_shape), 0), tf.int32)
        sample_shape = tf.reshape(
            sample_shape,
            tf.pad(tf.shape(sample_shape), paddings=[[pad, 0]], constant_values=1),
        )
        samples = sample_n_fn(tf.reduce_prod(sample_shape))
        batch_event_shape = tf.shape(samples)[1:]
        final_shape = tf.concat([sample_shape, batch_event_shape], 0)
        return tf.reshape(samples, final_shape)


def bernoulli_sample(
    probs=None, logits=None, dtype=tf.int32, sample_shape=(), seed=None
):
    """Samples from Bernoulli distribution."""
    if probs is None:
        probs = tf.sigmoid(logits, name="probs")
    else:
        probs = tf.convert_to_tensor(probs, name="probs")
    batch_shape_tensor = tf.shape(probs)

    def _sample_n(n):
        """Sample vector of Bernoullis."""
        new_shape = tf.concat([[n], batch_shape_tensor], 0)
        uniform = tf.random.uniform(new_shape, seed=seed, dtype=probs.dtype)
        return tf.cast(tf.less(uniform, probs), dtype)

    return _call_sampler(_sample_n, sample_shape)


def categorical_sample(logits, dtype=tf.int32, sample_shape=(), seed=None):
    """Samples from categorical distribution."""
    logits = tf.convert_to_tensor(logits, name="logits")
    event_size = tf.shape(logits)[-1]
    batch_shape_tensor = tf.shape(logits)[:-1]

    def _sample_n(n):
        """Sample vector of categoricals."""
        if logits.shape.ndims == 2:
            logits_2d = logits
        else:
            logits_2d = tf.reshape(logits, [-1, event_size])
        sample_dtype = tf.int64 if logits.dtype.size > 4 else tf.int32
        draws = tf.random.categorical(logits_2d, n, dtype=sample_dtype, seed=seed)
        draws = tf.reshape(tf.transpose(draws), tf.concat([[n], batch_shape_tensor], 0))
        return tf.cast(draws, dtype)

    return _call_sampler(_sample_n, sample_shape)

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
                samp = TrainingSamp()
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
                samp = NucleusSampler()
                infer_decoder = tfa.seq2seq.BasicDecoder(cell=self.decoder_cell, sampler=samp,
                                                         output_layer=self.output_dense)
                infer_output, _, _ = tfa.seq2seq.dynamic_decode(
                    decoder=infer_decoder,
                    swap_memory=True,
                    maximum_iterations=args['max_len'] + 1,
                    decoder_init_input=self.embedding,
                    decoder_init_kwargs={
                        'start_tokens': tf.tile(tf.constant([args['SOS_ID']], dtype=tf.int32), [tf.shape(context_with_latent)[1]]),
                        'end_token': args['EOS_ID'],
                        'initial_state': init_state_tuple
                    })
                # infer_predicted_ids = infer_output.predicted_ids[:, :, 0]  # select the first sentence
                infer_predicted_ids = infer_output.sample_id
                # The return is a list consisting of only one element whose diamention is (batch_size, max_len)
                return infer_predicted_ids