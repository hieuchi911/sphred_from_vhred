from configs import args
from model import VHRED
from data import VHREDDataLoader, ids_to_words

import tensorflow as tf
import numpy as np


class VHREDTester(object):

    def main(self):
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        VHRED_dl = VHREDDataLoader(sess, realtime=True)
        VHRED_model = VHRED(dataLoader=VHRED_dl)
        init_op = tf.compat.v1.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver()

        sess.run(init_op) # Initialize all variables declared, this let the variables really
        
        self.realtime_test(VHRED_model, VHRED_dl, sess)
        sess.close()


    def realtime_test(self, model, dataLoader, sess): # initials is the first 2 utterances (ids), user will input the third, model generate 4th response
        model.load(self.saver, sess, args['vhred_ckpt_dir'])
        utt = []
        print("Please create context for the conversation with 3 initial utterances:\n")
        for i in range(3):
            utt.append(input("\t\t"))
        utterances = " __eot__ ".join(utt)
        enc_inp_realtime_test = dataLoader.set_test_data(utterances)
        infer_decoder_ids_general = model.infer_decoder_session(sess, enc_inp_realtime_test)
        resp = ids_to_words(infer_decoder_ids_general[0], dataLoader.id_to_word, is_pre_utterance=False)
        print("Processing...")
        print("\t\tVHRED Bot:\t", resp)
        utt.append(resp)
        while True:
            user_res = input("\t\tYou:\t")
            if user_res == "<<exit>>":
                break
            utt.append(user_res)
            utterances = " __eot__ ".join(utt[-3:])
            enc_inp_realtime_test = dataLoader.set_test_data(utterances)
            infer_decoder_ids_general = model.infer_decoder_session(sess, enc_inp_realtime_test)
            resp = ids_to_words(infer_decoder_ids_general[0], dataLoader.id_to_word, is_pre_utterance=False)
            print("\t\tVHRED Bot:\t", resp)
            utt.append(resp)
