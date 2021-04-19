from configs import args
from model import Encoder, VHRED
from data import VHREDDataLoader, ids_to_words
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf
import datetime
import math
import numpy as np


class VHREDTrainer(object):

    def main(self):
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        VHRED_dl = VHREDDataLoader(sess)
        VHRED_model = VHRED(dataLoader=VHRED_dl)
        init_op = tf.compat.v1.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver()

        sess.run(init_op) # Initialize all variables declared, this let the variables really
        # set their values to specified values feed (very much like compiling). Only after this do other operations can be done
        # on these variables (updating, backpropagating etc.)

        # loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=True)
        loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=False)

        sess.close()

    def sample_test(self, model, dataLoader, sess):
        for enc_inp, dec_inp, dec_tar in dataLoader.test_generator():
            infer_decoder_ids = model.infer_decoder_session(sess, enc_inp)
            
            sample_previous_utterance_id = enc_inp[:3]
            sample_infer_response_id = infer_decoder_ids[-1][:3]
            sample_true_response_id = dec_tar[:3]
            for i in range(len(sample_infer_response_id)):
                print('-----------------------------------')
                print('previous utterances:')
                print(ids_to_words(sample_previous_utterance_id[i], dataLoader.id_to_word, is_pre_utterance=True))
                print('true response:')
                print(ids_to_words(sample_true_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
                print('infer response:')
                print(ids_to_words(sample_infer_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
                print('-----------------------------------')
            break

    def train_model(self, model, dataLoader, sess, is_fresh_model=True):
        if not is_fresh_model:
            model.load(self.saver, sess, args['vhred_ckpt_dir'])
            # current_lr = sess.run(model.lr)
            # sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 5})
        best_result_loss = 1000.0
        stop = False
        best_loss = 1000
        last_improvement = 0
        loss_list = sess.run(model.loss_list)
        test_loss_list = sess.run(model.test_loss_list)
        training_epoch_loss = sess.run(model.training_epoch_loss)
        validation_epoch_loss = sess.run(model.validation_epoch_loss)
        model_epoch = sess.run(model.epoch)
        for epoch in range(args['n_epochs']):
            x = np.arange(model_epoch+1)
            plt.plot(training_epoch_loss)
            plt.plot(validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper left')
            plt.show()

            plt.plot(loss_list)
            plt.show()
            if stop:
              print('\n\nlast_improvement is: ', last_improvement)
              print("No improvements so cease training\n\n")
              break
            print()
            print("---- epoch: {}/{} | lr: {} ----".format(epoch, args['n_epochs'], sess.run(model.lr)))
            tic = datetime.datetime.now()

            train_batch_num = dataLoader.train_batch_num
            test_batch_num = dataLoader.test_batch_num

            loss = 0.0
            nll_loss = 0.0
            kl_loss = 0.0
            loss_bf_update = 0.0
            kl_loss_bf_update = 0.0
            loss_aft_update = 0.0
            kl_loss_aft_update = 0.0
            count = 0
            the_line = "------------------------------------------------------------"

            for (enc_inp, dec_inp, dec_tar) in tqdm(dataLoader.train_generator(), desc="training"):
                count += 1
                # loss_bf_update += model.loss_session(sess, enc_inp, dec_inp, dec_tar)
                # kl_loss_bf_update += model.kl_loss_session(sess, enc_inp, dec_inp)
                # if count % args['display_step'] == 0:
                #   print(f"\n\n\n\n\n\n{the_line}\n{the_line}\nBEFORE UPDATING TRAINABLE VARIABLES\n{the_line}\n{the_line}\n\n\n\n\n\n")
                #   print("For enc_inp: ", ids_to_words(enc_inp[0], dataLoader.id_to_word, is_pre_utterance=True))
                #   print("The encoded hidden vectors for the 3 previous utterances above is: ", model.encoder_state_session(sess, enc_inp))
                #   print("\nThe context 2 hidden vectors for these utterances is: ", model.context_state_session(sess, enc_inp))
                #   print("The correct 4th response (dec_tar) is: ", ids_to_words(dec_tar[0], dataLoader.id_to_word, is_pre_utterance=False))
                #   print("The posterior latent z deducted from encoder RNN output of dec_tar and context is: ", model.posterior_z_session(sess, enc_inp, dec_inp, dec_tar))
                #   train_logits, train_sample_ids = model.train_decoder_session(sess, enc_inp, dec_inp, dec_tar)
                #   print("So the decoded logits (train_logits from TRAIN DECODER SESSION) is: ", train_logits)
                #   print("-----> These logits correspond to following ids (train_sample_id from TRAIN DECODER SESSION, will be compared with dec_tar):\n", train_sample_ids)
                #   print("Compare these above ids with true ids: (dec_tar)\n", dec_tar)
                #   print("Specifically compare: first train_sample_ids: \n\t\t", train_sample_ids[0], "\n\t\t and first dec_tar:\n\t\t", dec_tar[0])
                  
                #   print("\n\n==========\n\nTherefore, loss between decoded output and decoder target dec_tar is: ", loss_bf_update/count)
                #   print("Also, KL difference between posterior and prior distribution is: ", kl_loss_bf_update/count)
                #   print("\n\n==========\n\n")
                # print("\nENTER OF TRAIN SESSION\n")
                
                train_out = model.train_session(sess, enc_inp, dec_inp, dec_tar)
                
                # print("\nOUT OF TRAIN SESSION\n")
                # loss_aft_update += model.loss_session(sess, enc_inp, dec_inp, dec_tar)
                # kl_loss_aft_update += model.kl_loss_session(sess, enc_inp, dec_inp)
                # if count % args['display_step'] == 0:
                #   if count < 25:
                #     print("dec_inp: \n", dec_inp)
                #   print(f"\n\n\n\n\n\n{the_line}\n{the_line}\nAFTER UPDATING TRAINABLE VARIABLES\n{the_line}\n{the_line}\n\n\n\n\n\n")
                #   print("For enc_inp: ", ids_to_words(enc_inp[0], dataLoader.id_to_word, is_pre_utterance=True))
                #   print("The encoded hidden vectors for the 3 previous utterances above is: ", model.encoder_state_session(sess, enc_inp))
                #   print("\nThe context 2 hidden vectors for these utterances is: ", model.context_state_session(sess, enc_inp))
                #   print("The correct 4th response (dec_tar) is: ", ids_to_words(dec_tar[0], dataLoader.id_to_word, is_pre_utterance=False))
                #   print("The posterior latent z deducted from encoder RNN output of dec_tar and context is: ", model.posterior_z_session(sess, enc_inp, dec_inp, dec_tar))
                #   train_logits, train_sample_ids = model.train_decoder_session(sess, enc_inp, dec_inp, dec_tar)
                #   print("So the decoded logits (train_logits from TRAIN DECODER SESSION) is: ", train_logits)
                #   print("-----> These logits correspond to following ids (train_sample_id from TRAIN DECODER SESSION, will be compared with dec_tar):\n", train_sample_ids)
                #   print("Compare previous ids with true ids: (dec_tar)\n", dec_tar)
                #   print("Specifically compare: first train_sample_ids: \n\t\t", train_sample_ids[0], "\n\t\t and first dec_tar:\n\t\t", dec_tar[0])
                #   print("\n\n==========\n\nTherefore, loss between decoded output and decoder target dec_tar is: ", loss_aft_update/count)
                #   print("Also, KL difference between posterior and prior distribution is: ", kl_loss_aft_update/count)
                #   print("\n\n==========\n\n")
                
              
                
                global_step = train_out['global_step']
                loss += train_out['loss']
                nll_loss += train_out['nll_loss']
                kl_loss += train_out['kl_loss']
                
                # if last_improvement > 30 and last_improvement < 60:
                #     current_lr = sess.run(model.lr)
                #     sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 1.25})
                #     print("\n\n******Updated learning rate by 1.25 from ", current_lr, " to ",current_lr*1.25, "******\n\n")
                # elif last_improvement >= 60 and last_improvement < 80:
                #     # stop = True
                #     bar = '---------------------------------------------------'
                #     print("\n\n\n", bar, "\n", bar, "\n", bar, "\n", "stop is now True, last_improvement is: ", last_improvement, "\n", bar, "\n", bar, "\n", bar, "\n\n\n")
                #     # break
                # elif last_improvement >= 80 and last_improvement < 95:
                #     current_lr = sess.run(model.lr)
                #     sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 0.8})
                #     print("\n\n******Updated learning rate by 0.8 from ", current_lr, " to ",current_lr*0.8, "******\n\n")

                # This if block below should be placed outsite of the display step, since it
                # should verify on each train session performed on a data point
                current_loss = loss / count
                loss_list = np.append(loss_list, current_loss)
                if current_loss < best_loss:
                  best_loss = current_loss
                  last_improvement = 0
                  stop = False
                else:
                  last_improvement += 1
                #   print("\n\nNo improvement, best loss is: ", best_loss, "last improvement: ", last_improvement ,"\n\n")
                if count % args['display_step'] == 0:
                    if current_loss < best_loss:
                        print("\n\nImproved from", best_loss, "to ", current_loss, "\n\n")
                    else:
                        print("\n\nNo improvement, best loss is: ", best_loss, "last improvement: ", last_improvement ,"\n\n")
                    model.train_decoder_session(sess, enc_inp, dec_inp, dec_tar)
                    current_loss = loss / count
                    current_nll_loss = nll_loss / count
                    current_kl_loss = kl_loss / count
                    current_perplexity = math.exp(float(current_nll_loss)) if current_nll_loss < 300 else float("inf")
                    print('Step {} | Batch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(global_step,
                                                                                                       count,
                                                                                                       train_batch_num,
                                                                                                       current_loss,
                                                                                                       current_nll_loss,
                                                                                                       current_kl_loss,
                                                                                                       current_perplexity))
                    # length_of_enc_state_list, states = model.encoder_state_session(sess, enc_inp)
                    # print("states: ", np.shape(states))
                    # print("length_of_enc_state_list: ", np.shape(length_of_enc_state_list))
            # print("\n\n\n*******************************trainable variables: ", len(sess.run(model.tvars)), "\n************************************\n\n\n")
            # print("latent plus context output: ", sess.run(model.context_with_latent_infer))
            training_epoch_loss = np.append(training_epoch_loss, np.mean(loss_list))


            print(count)
            loss = loss / count
            nll_loss = nll_loss / count
            kl_loss = kl_loss / count
            perplexity = math.exp(float(nll_loss)) if nll_loss < 300 else float("inf")
            print('Train Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(epoch,
                                                                                           args['n_epochs'],
                                                                                           loss,
                                                                                           nll_loss,
                                                                                           kl_loss,
                                                                                           perplexity))

            print("Loss plot per iteration is: ")
            plt.plot(loss_list)
            plt.show()
            
            test_loss = 0.0
            test_nll_loss = 0.0
            test_kl_loss = 0.0
            test_count = 0
            # test_count = 1
            for (enc_inp, dec_inp, dec_tar) in tqdm(dataLoader.test_generator(), desc="testing"):
                test_out = model.test_session(sess, enc_inp, dec_inp, dec_tar)
                test_loss += test_out['loss_test']
                test_nll_loss += test_out['nll_loss_test']
                test_kl_loss += test_out['kl_loss']
                test_count += 1
                test_loss_list = np.append(test_loss_list, test_out['loss_test'])
                if stop and epoch < 2:
                    break
            test_loss /= test_count
            test_nll_loss /= test_count
            test_kl_loss /= test_count
            test_perplexity = math.exp(float(test_nll_loss)) if test_nll_loss < 300 else float("inf")
            print('Test Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(epoch,
                                                                                          args['n_epochs'],
                                                                                          test_loss,
                                                                                          test_nll_loss,
                                                                                          test_kl_loss,
                                                                                          test_perplexity))
            validation_epoch_loss = np.append(validation_epoch_loss, np.mean(test_loss_list))
            print("Training loss vs. Testing loss per epoch is: ")
            x = np.arange(model_epoch+1)
            plt.plot(x, training_epoch_loss)
            plt.plot(x, validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper left')
            plt.show()

            print()

            print('# sample test')
            self.sample_test(model, dataLoader, sess)
            
            sess.run(model.update_epoch_op, feed_dict={model.new_epoch: model_epoch})
            sess.run(model.update_loss_list_op, feed_dict={model.new_loss_list: loss_list})
            sess.run(model.update_test_loss_list_op, feed_dict={model.new_test_loss_list: test_loss_list})
            sess.run(model.update_training_epoch_loss_list_op, feed_dict={model.new_training_epoch_loss_list: training_epoch_loss})
            sess.run(model.update_validation_epoch_loss_list_op, feed_dict={model.new_validation_epoch_loss_list: validation_epoch_loss})
            
            # model.save(self.saver, sess, args['vhred_ckpt_dir'])
            if test_loss < best_result_loss:
                print("Save model since test_loss", test_loss, " < best_result_loss", best_result_loss)
                model.save(self.saver, sess, args['vhred_ckpt_dir'])
                if np.abs(best_result_loss - test_loss) < 0.01:
                    current_lr = sess.run(model.lr)
                    # model.update_lr_op is an operation that assign model.lr with model.new_lr
                    sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 0.95})
                    print("\n\n******Decreased learning rate by 0.95 from ", current_lr, " to ",current_lr*0.95, "******\n\n")
                best_result_loss = test_loss
            toc = datetime.datetime.now()
            print(" # Epoch finished in {}".format(toc - tic))
            model_epoch += 1
        print("Loss plot per iteration is: ")
        plt.plot(loss_list)
        plt.show()
        print("Training loss vs. Testing loss per epoch is: ")
        x = np.arange(model_epoch+1)
        plt.plot(x, training_epoch_loss)
        plt.plot(x, validation_epoch_loss)
        plt.legend(['training loss', 'validation loss'], loc='upper left')
        plt.show()