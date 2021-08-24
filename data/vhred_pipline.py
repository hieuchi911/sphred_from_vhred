from data.data_process import read_data, read_vocab, tokenize_data, split_data, form_input_data, word_embeddings
from configs import args
import numpy as np

class VHREDDataLoader(object):
    def __init__(self, sess, realtime=False, utterances=None, flag=None):
        self.sess = sess

        if realtime:
            self.word_to_id, self.id_to_word = read_vocab(filename='Ubuntu_vocab.txt')

            self.vocab_size = len(self.word_to_id)

            self.embedding_matrix = word_embeddings(self.word_to_id.keys())

        else:
            #load data
            # self.raw_data, self.labels = read_data(filename='Ubuntu_no_punct.txt')
            # self.word_to_id, self.id_to_word = read_vocab(filename='Ubuntu_vocab.txt')

            self.raw_data, self.labels = read_data(filename='Squad_no_punct.txt')
            self.word_to_id, self.id_to_word = read_vocab(filename='Squad_vocab.txt')
           
            self.vocab_size = len(self.word_to_id)
            print("vocab_size: ", self.vocab_size)
            
            self.data = tokenize_data(self.raw_data, self.word_to_id)
            self.embedding_matrix = word_embeddings(self.word_to_id.keys())

            X_train, y_train, X_test, y_test, self.x_labels, self.y_labels, self.x_test_labels, self.y_test_labels, self.y_test_labels_general = split_data(self.data, self.labels)
            self.train_size = len(X_train)
            self.test_size = len(X_test)
            self.train_batch_num = self.train_size // args['batch_size']
            self.test_batch_num = self.test_size // args['batch_size']


            # encoder_input, decoder_input, decoder_output
            self.enc_inp_train, self.dec_inp_train, self.dec_out_train = form_input_data(X=X_train, y=y_train)
            self.enc_inp_test, self.dec_inp_test, self.dec_out_test = form_input_data(X=X_test, y=y_test)
            
            # print(self.enc_inp_test[0:6])
    
    def set_test_data(self, utterances, flag):
        self.raw_data, self.labels = read_data(utterances=utterances, flag=flag)
        enc_inp_realtime_test = tokenize_data(self.raw_data, self.word_to_id)   # (1, 3 utterance, 15 words), this will be fed into the encoder
        test_inp = form_input_data(realtime_test=True, dialogue=enc_inp_realtime_test)
        return test_inp, self.labels

    def train_generator(self):
        for i in range(self.train_batch_num):
            start_index = i * args['batch_size']
            end_index = min((i + 1) * args['batch_size'], self.train_size)
            encoder_inputs = self.enc_inp_train[start_index: end_index]
            decoder_inputs = self.dec_inp_train[start_index: end_index]
            decoder_targets = self.dec_out_train[start_index: end_index]
            
            x_labels = self.x_labels[start_index: end_index]
            y_labels = self.y_labels[start_index: end_index]
            yield encoder_inputs, decoder_inputs, decoder_targets, x_labels, y_labels

    def test_generator(self):
        for i in range(self.test_batch_num):
            start_index = i * args['batch_size']
            end_index = min((i + 1) * args['batch_size'], self.test_size)
            encoder_inputs = self.enc_inp_test[start_index: end_index]
            decoder_inputs = self.dec_inp_test[start_index: end_index]
            decoder_outputs = self.dec_out_test[start_index: end_index]

            x_labels = self.x_test_labels[start_index: end_index]
            y_labels_general = self.y_test_labels_general[start_index: end_index]
            y_labels = self.y_test_labels[start_index: end_index]
            yield encoder_inputs, decoder_inputs, decoder_outputs, x_labels, y_labels, y_labels_general


if __name__ == '__main__':
    loader = VHREDDataLoader(sess='as')
    print('--------------------------------------')
    print(loader.dec_inp_train.shape)
    print(loader.dec_inp_train[0].shape)
    print(loader.dec_out_train[:2])
    print(loader.dec_inp_train[:2])
    print(loader.dec_inp_train[0])

