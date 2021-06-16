from configs import args
from model import Encoder, VHRED
from data import VHREDDataLoader, ids_to_words
import matplotlib.pyplot as plt

from tqdm import tqdm
import tensorflow as tf
import datetime
import math
import numpy as np

import time
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns


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

        loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=True)
        # loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=False)
        # self.sample_test(VHRED_model, VHRED_dl, sess, True)

        sess.close()

    def sample_test(self, model, dataLoader, sess, inference=True):
        if inference:
            model.load(self.saver, sess, args['vhred_ckpt_dir'])
            training_epoch_loss = sess.run(model.training_epoch_loss)
            validation_epoch_loss = sess.run(model.validation_epoch_loss)
            print("Last training loss: ", training_epoch_loss[-1])
            print("Last test loss: ", validation_epoch_loss[-1])
            plt.plot(training_epoch_loss)
            plt.plot(validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper left')
            plt.show()
        for enc_inp, dec_inp, dec_tar, x_labels, y_labels, y_labels_general in dataLoader.test_generator():
            infer_decoder_ids_general = model.infer_decoder_session(sess, enc_inp, x_labels, y_labels_general)
            infer_decoder_ids = model.infer_decoder_session(sess, enc_inp, x_labels, y_labels)

            sample_previous_utterance_id = enc_inp[:10]
            sample_infer_response_id_general = infer_decoder_ids_general[-1][:10]
            sample_infer_response_id = infer_decoder_ids[-1][:10]
            sample_true_response_id = dec_tar[:10]

            X_prior = np.empty((0, args['latent_size']), float)
            X_post = np.empty((0, args['latent_size']), float)

            for i in range(len(sample_infer_response_id)):
                print('-----------------------------------')
                print('previous utterances:')
                print(ids_to_words(sample_previous_utterance_id[i], dataLoader.id_to_word, is_pre_utterance=True))
                print('true response:')
                print(ids_to_words(sample_true_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
                print('infer general response:')
                print(ids_to_words(sample_infer_response_id_general[i], dataLoader.id_to_word, is_pre_utterance=False))
                print('infer detailed response:')
                print(ids_to_words(sample_infer_response_id[i], dataLoader.id_to_word, is_pre_utterance=False))
                print('-----------------------------------')
            if inference:
                prior_z = model.prior_z_session(sess, enc_inp, x_labels)
                post_z = model.posterior_z_session(sess, enc_inp, dec_tar, x_labels)
                X_prior = np.concatenate((X_prior, prior_z[0]))
                X_post = np.concatenate((X_post, post_z[0]))
            else:
                break
        if inference:
            X = np.concatenate((X_prior, X_post))
            feat_cols = ['latent'+str(i) for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feat_cols)
            label = np.concatenate((np.full([X_prior.shape[0]], 'prior'), np.full([X_post.shape[0]], 'post')))
            df['label'] = label

            np.random.seed(42)
            rndperm = np.random.permutation(df.shape[0])

            print('Size of the dataframe: {}'.format(df.shape)) # Expect: test_size, 64
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df[feat_cols].values)
            df['pca-one'] = pca_result[:,0]
            df['pca-two'] = pca_result[:,1] 
            df['pca-three'] = pca_result[:,2]
            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

            sns.scatterplot(
              x="pca-one", y="pca-two",
              hue="label",
              palette=sns.color_palette("hls", 2),
              data=df.loc[rndperm,:],
              legend="full",
              alpha=0.3
            )
            plt.show()

    def train_model(self, model, dataLoader, sess, is_fresh_model=True):
        if not is_fresh_model:
            model.load(self.saver, sess, args['vhred_ckpt_dir'])
            # current_lr = sess.run(model.lr)
            # sess.run(model.update_lr_op, feed_dict={model.new_lr: current_lr * 5})
        best_result_loss = sess.run(model.best_test_loss)
        stop = False
        best_loss = sess.run(model.best_loss)
        last_improvement = sess.run(model.improve)
        loss_list = sess.run(model.loss_list)
        nll_loss_list = sess.run(model.nll_loss_list)
        kl_loss_list = sess.run(model.kl_loss_list)
        kl_weight_list = sess.run(model.kl_weight_list)
        test_loss_list = sess.run(model.test_loss_list)
        training_epoch_loss = sess.run(model.training_epoch_loss)
        validation_epoch_loss = sess.run(model.validation_epoch_loss)
        model_epoch = sess.run(model.epoch)
        for epoch in range(args['n_epochs']):
            if last_improvement > 7:
              print('\n\nlast_improvement is: ', last_improvement)
              print("No improvements so cease training\n\n")
              break
            print()
            print("---- epoch: {}/{} | lr: {} ----".format(model_epoch, args['n_epochs'], sess.run(model.lr)))
            tic = datetime.datetime.now()

            train_batch_num = dataLoader.train_batch_num
            test_batch_num = dataLoader.test_batch_num

            loss = 0.0
            nll_loss = 0.0
            kl_loss = 0.0
            kl_weight = 0.0
            loss_bf_update = 0.0
            kl_loss_bf_update = 0.0
            loss_aft_update = 0.0
            kl_loss_aft_update = 0.0
            count = 0
            X_prior = np.empty((0, args['latent_size']), float)
            X_post = np.empty((0, args['latent_size']), float)

            for (enc_inp, dec_inp, dec_tar, x_labels, y_labels) in tqdm(dataLoader.train_generator(), desc="training"):
                count += 1
                
                train_out = model.train_session(sess, enc_inp, dec_inp, dec_tar, x_labels, y_labels)
                
                global_step = train_out['global_step']
                loss += train_out['loss']
                nll_loss += train_out['nll_loss']
                kl_loss += train_out['kl_loss']
                kl_weight += train_out['kl_weights']
                
                current_loss = loss / count
                current_nll_loss = nll_loss / count
                current_kl_loss = kl_loss / count
                current_kl_weight = kl_weight / count
                loss_list = np.append(loss_list, current_loss)
                nll_loss_list = np.append(nll_loss_list, current_nll_loss)
                kl_loss_list = np.append(kl_loss_list, current_kl_loss)
                kl_weight_list = np.append(kl_weight_list, train_out['kl_weights'])
                
                if count % args['display_step'] == 0:
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
                    prior_z = model.prior_z_session(sess, enc_inp, x_labels)
                    post_z = model.posterior_z_session(sess, enc_inp, dec_tar, x_labels)
                    X_prior = np.concatenate((X_prior, prior_z[0]))
                    X_post = np.concatenate((X_post, post_z[0]))
            # print("\n\n\n*******************************trainable variables: ", len(sess.run(model.tvars)), "\n************************************\n\n\n")
            # print("latent plus context output: ", sess.run(model.context_with_latent_infer))
            training_epoch_loss = np.append(training_epoch_loss, np.mean(loss_list))

            loss = loss / count
            nll_loss = nll_loss / count
            kl_loss = kl_loss / count
            perplexity = math.exp(float(nll_loss)) if nll_loss < 300 else float("inf")
            print('Train Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(model_epoch,
                                                                                           args['n_epochs'],
                                                                                           loss,
                                                                                           nll_loss,
                                                                                           kl_loss,
                                                                                           perplexity))
            X = np.concatenate((X_prior, X_post))
            feat_cols = ['latent'+str(i) for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=feat_cols)
            label = np.concatenate((np.full([X_prior.shape[0]], 'prior'), np.full([X_post.shape[0]], 'post')))
            df['label'] = label

            np.random.seed(42)
            rndperm = np.random.permutation(df.shape[0])

            print('Size of the dataframe: {}'.format(df.shape)) # Expect: test_size, 64
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(df[feat_cols].values)
            df['pca-one'] = pca_result[:,0]
            df['pca-two'] = pca_result[:,1] 
            df['pca-three'] = pca_result[:,2]
            print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

            sns.scatterplot(
              x="pca-one", y="pca-two",
              hue="label",
              palette=sns.color_palette("hls", 2),
              data=df.loc[rndperm,:],
              legend="full",
              alpha=0.3
            )
            plt.show()

            print("Loss plot per iteration is: ")
            plt.plot(loss_list)
            plt.legend(['loss'], loc='upper right')
            plt.show()

            print("Negative log likelihood (nll) loss vs KL loss plot per iteration is: ")
            plt.plot(nll_loss_list)
            plt.plot(kl_loss_list)
            plt.legend(['nll loss', 'kl loss'], loc='upper right')
            plt.show()

            print("KL loss plot vs KL loss weight per iteration is: ")
            plt.plot(kl_loss_list)
            plt.plot(kl_weight_list)
            plt.legend(['kl loss', 'kl_weight'], loc='upper right')
            plt.show()

            if loss < best_loss:
              best_loss = loss
              last_improvement = 0
            else:
              last_improvement += 1
            
            test_loss = 0.0
            test_nll_loss = 0.0
            test_kl_loss = 0.0
            test_count = 0
            # test_count = 1
            for (enc_inp, dec_inp, dec_tar, x_labels, y_labels, y_labels_general) in tqdm(dataLoader.test_generator(), desc="testing"):
                test_out = model.test_session(sess, enc_inp, dec_inp, dec_tar, x_labels, y_labels)
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
            print('Test Epoch {}/{} | Loss {} | NLL_loss {} | KL_loss {} | PPL {}'.format(model_epoch,
                                                                                          args['n_epochs'],
                                                                                          test_loss,
                                                                                          test_nll_loss,
                                                                                          test_kl_loss,
                                                                                          test_perplexity))
            validation_epoch_loss = np.append(validation_epoch_loss, np.mean(test_loss_list))
            print("Training loss vs. Testing loss per epoch is: ")
            plt.plot(training_epoch_loss)
            plt.plot( validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper right')
            plt.show()

            print()

            print('# sample test')
            self.sample_test(model, dataLoader, sess, inference=False)
            
            sess.run(model.update_epoch_op, feed_dict={model.new_epoch: model_epoch})
            sess.run(model.update_best_loss_op, feed_dict={model.new_best_loss: best_loss})
            sess.run(model.update_improve_op, feed_dict={model.new_improve: last_improvement})

            
            sess.run(model.update_loss_list_op, feed_dict={model.new_loss_list: loss_list})
            sess.run(model.update_nll_loss_list_op, feed_dict={model.new_nll_loss_list: nll_loss_list})
            sess.run(model.update_kl_loss_list_op, feed_dict={model.new_kl_loss_list: kl_loss_list})
            sess.run(model.update_kl_weight_list_op, feed_dict={model.new_kl_weight_list: kl_weight_list})
            sess.run(model.update_test_loss_list_op, feed_dict={model.new_test_loss_list: test_loss_list})
            sess.run(model.update_training_epoch_loss_list_op, feed_dict={model.new_training_epoch_loss_list: training_epoch_loss})
            sess.run(model.update_validation_epoch_loss_list_op, feed_dict={model.new_validation_epoch_loss_list: validation_epoch_loss})
            
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
            sess.run(model.update_best_test_loss_op, feed_dict={model.new_best_test_loss: best_result_loss})
            if last_improvement <= 7:
                model.save(self.saver, sess, args['vhred_ckpt_dir'])
            print(" # Epoch finished in {}".format(toc - tic))
            model_epoch += 1
        # print("Loss plot per iteration is: ")
        # plt.plot(loss_list)
        # plt.show()
        # print("Training loss vs. Testing loss per epoch is: ")
        # plt.plot(training_epoch_loss)
        # plt.plot(validation_epoch_loss)
        # plt.legend(['training loss', 'validation loss'], loc='upper left')
        # plt.show()