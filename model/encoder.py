from base import BaseModel
from configs import args
from model.model_utils import create_multi_rnn_cell
import numpy as np

import tensorflow as tf

# encoder_RNN, context_RNN
class Encoder(BaseModel):
    def __init__(self, vocab_size, word_embeddings=None, is_embedding=True):
        # if is_embedding=True, this is encoder_RNN
        # if is_embedding=False, this is context_RNN
        super().__init__('encoder')
        self.is_embedding = is_embedding
        # variable_scope: a context manager, helps differentiate different variables. get_variable is used to define new
        # vars, also to retrieve the required var if it existed before (the retrieval is enabled with reuse set to True/ like below)
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            # Create an embedding, named 'lookup_table' that is a lookup table, with shape of [vocab_size, embed_dims] where
            # embed_dims is the length of each word embedding? The first dimension refers to a word's index in the vocabulary
            # while the second dimension is its embedding with length of 128
            if self.is_embedding:
              self.embedding = tf.constant(word_embeddings, dtype=tf.float32)
            # self.embedding = tf.compat.v1.get_variable('lookup_table', [vocab_size, args['embed_dims']])
            
            # The cell is a processing unit, it has a state (hidden state), takes as input embeddings and previous state
            # create_multi_rnn_cell args are: 1 - cell type (LSTM/ GRU), 2 - hidden state length, 3 - keep probability (for dropout
            # mechanism of keeping only a certain amount of input, output data), 4 - number of layers
            self.encoder_cell = create_multi_rnn_cell(args['rnn_type'], args['rnn_size'], args['keep_prob'],
                                                      args['num_layer'])    # multi-layer RNN will have its intermediate layers
            # taking output of previous layer and previous time step hidden state as its input (stacked RNN)
            # *** Stacked RNN has ability to induce representations at differing levels of abstraction across layers
            # *** Initial layers of stacked networks can induce representations that serve as useful abstractions for further layers

    def __call__(self, inputs):
        with tf.compat.v1.variable_scope(self._scope, reuse=tf.compat.v1.AUTO_REUSE):
            if self.is_embedding:
                # Used for Encoder RNN, since this use meaningful embedding from inputs, it must turn index-based input to
                # more meaningful embedding (like wordvec), it has to use the embedding_lookup function to turn index-based
                # into word2vec embeddings
                """ count number of non-zeros in the input tensor (the input is the encoded utterances like: [89  37   4
				 27 358   7  19  13   3   2   0   0   0   0   0   0] where 0's are result of padding)"""
                encoder_lengths = tf.math.count_nonzero(inputs, -1, dtype=tf.int32)
                # it's like indexing in python list, args: 1 - params: the vocabulary, size is [vocab_len, embedding_len]
                # the first dimension is the index of each vocab word, second dimension is the embedding of each word; 2 - ids: the id
                # list of the words in the input
                embedded_inputs = tf.nn.embedding_lookup(params=self.embedding, ids=inputs)
            else:
                # Used for Context RNN, and since it uses Encoder RNN output, which is already a meaningful embedding (non-index style)
                encoder_lengths = None
                embedded_inputs = inputs

            # plug in the embedded_inputs above (context/ encoder RNN) to the RNN cell defined in self.encoder_cell of create_multi_rnn_cell
            outputs, states = tf.compat.v1.nn.dynamic_rnn(self.encoder_cell, embedded_inputs, encoder_lengths,
                                                          dtype=tf.float32)
            return outputs, states


# if __name__ == '__main__':
#     sess = tf.compat.v1.Session()
#     enc_inputs = np.array([[2, 3, 4], [3, 4, 5]])
#     vocab_size = 18506
#     encoder = Encoder(vocab_size)
#     enc_inp_ph = tf.compat.v1.placeholder(tf.int32, [None, None])
#     outputs, states = encoder(enc_inp_ph)

#     # with tf.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())
#     result = sess.run(states, feed_dict={enc_inp_ph: enc_inputs})
#     print(type(result))
#     print(len(result))
#     print(result[-1].shape)
#     print(type(result[-1]))
