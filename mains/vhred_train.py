from configs import args
from model import Encoder, VHRED
from data import VHREDDataLoader, ids_to_words
from .mains_helper import embedding_eval, eval_emb_metrics
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
import json

class VHREDTrainer(object):

    def main(self, is_nucleus=True):
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        VHRED_dl = VHREDDataLoader(sess)
        VHRED_model = VHRED(dataLoader=VHRED_dl, is_nucleus=is_nucleus)
        init_op = tf.compat.v1.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver()

        sess.run(init_op) # Initialize all variables declared, this let the variables really
        # set their values to specified values feed (very much like compiling). Only after this do other operations can be done
        # on these variables (updating, backpropagating etc.)

        # UNCOMMENT THIS TO TRAIN NEW MODEL
        # loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=True)
        
        # UNCOMMENT THIS TO CONTINUE TRAINING TRAINED MODEL
        # loss_list = self.train_model(VHRED_model, VHRED_dl, sess, is_fresh_model=False)
        
        # UNCOMMENT THIS TO TEST TRAINED MODEL
        self.sample_test(VHRED_model, VHRED_dl, sess, True)

        sess.close()
    
    def draw_scales(self, dmode, list1, name1, list2, name2):
        # More versatile wrapper
        fig, host = plt.subplots(figsize=(6,4)) # (width, height) in inches

        par1 = host.twinx()
            
        host.set_xlim(0 - len(list1) * 0.1, len(list1)-1)
        host.set_ylim(0 - max(list1) * 0.02, max(list1) + 0.3*max(list1))
        par1.set_ylim(0 - max(list2) * 0.02, max(list2) + 0.3*max(list2))
            
        host.set_xlabel('iteration')
        host.set_ylabel(name1 + "(" + dmode + " average)")
        par1.set_ylabel(name2 + "(" + dmode + " average)")

        color1 = plt.cm.viridis(0)
        color2 = plt.cm.viridis(0.5)
        color3 = plt.cm.viridis(.9)

        p1, = host.plot(range(len(list1)), list1, color=color1, label=name1)
        p2, = par1.plot(range(len(list2)), list2, color=color2, label=name2)

        lns = [p1, p2]
        host.legend(handles=lns, loc='best')

        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())

        # Adjust spacings w.r.t. figsize
        fig.tight_layout()
        plt.show()

        # Best for professional typesetting, e.g. LaTeX
        # plt.savefig("pyplot_multiple_y-axis.pdf")
        # For raster graphics use the dpi argument. E.g. '[...].png", dpi=200)'


    def sample_test(self, model, dataLoader, sess, inference=True):
        if inference:
            model.load(self.saver, sess, args['vhred_ckpt_dir'])
            training_epoch_loss = sess.run(model.training_epoch_loss)
            validation_epoch_loss = sess.run(model.validation_epoch_loss)

            loss_list = sess.run(model.loss_list)
            nll_loss_list = sess.run(model.nll_loss_list)
            kl_loss_list = sess.run(model.kl_loss_list)        
            kl_weight_list = sess.run(model.kl_weight_list)

            print("Negative log likelihood (nll) loss vs KL loss plot per iteration is: ")
            self.draw_scales('iteration', nll_loss_list, 'NLL', kl_loss_list, 'KL')    

            print("KL loss plot vs KL loss weight per iteration is: ")
            self.draw_scales('iteration', kl_loss_list, 'KL', kl_weight_list, 'KL weight')
            
            print("Training loss vs. Testing loss per epoch is: ")
            print("Last training loss: ", training_epoch_loss[-1], "Last test loss: ", validation_epoch_loss[-1])
            plt.plot(training_epoch_loss)
            plt.plot(validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper left')
            plt.show()
        
        all_infer_ids = np.empty((0, args["max_len"]))
        all_target_ids = np.empty((0, args["max_len"]))
        
        X_prior = np.empty((0, args['latent_size']), float)
        X_post = np.empty((0, args['latent_size']), float)

        # all_general_response = []
        all_detailed_response = []
        all_ground_truth = []

        for enc_inp, dec_inp, dec_tar, x_labels, y_labels, y_labels_general in dataLoader.test_generator():
            infer_decoder_ids_general = model.infer_decoder_session(sess, enc_inp, x_labels, y_labels_general)
            infer_decoder_ids = model.infer_decoder_session(sess, enc_inp, x_labels, y_labels)
            
            if not inference:
                sample_previous_utterance_id = enc_inp[:10]
                sample_infer_response_id_general = infer_decoder_ids_general[:10]
                sample_infer_response_id = infer_decoder_ids[:10]
                sample_true_response_id = dec_tar[:10]

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
            else:
                for i in range(len(infer_decoder_ids_general)):
                    # # print('infer general response:')
                    # general = ids_to_words(infer_decoder_ids_general[i], dataLoader.id_to_word, is_pre_utterance=False)
                    # # print(general)
                    # all_general_response.append(general)
                    # print('infer detailed response:')
                    detail = ids_to_words(infer_decoder_ids[i], dataLoader.id_to_word, is_pre_utterance=False)
                    detail = detail.replace("<eos>", " ")
                    # print(detail)

                    ground_truth = ids_to_words(dec_tar[i], dataLoader.id_to_word, is_pre_utterance=False)
                    ground_truth = ground_truth.replace("<eos>", " ")

                    if len(detail.split()) < 1 or len(ground_truth.split()) < 1:
                        continue
                    
                    all_detailed_response.append(detail)
                    all_ground_truth.append(ground_truth)

                    # print('-----------------------------------')


            if inference:
                prior_z = model.prior_z_session(sess, enc_inp, x_labels)
                post_z = model.posterior_z_session(sess, enc_inp, dec_tar, x_labels)
                X_prior = np.concatenate((X_prior, np.mean(prior_z, axis=0)))
                X_post = np.concatenate((X_post, np.mean(post_z, axis=0)))
                try:
                    infer_decoder_ids = np.array([np.pad(m, (0, args["max_len"] - len(m))) for m in infer_decoder_ids])
                    all_target_ids = np.concatenate((all_target_ids, dec_tar))
                    all_infer_ids = np.concatenate((all_infer_ids, infer_decoder_ids))
                except:
                    print(np.shape(infer_decoder_ids))
                    print(infer_decoder_ids, "\n\n")
                    print(np.shape(dec_tar))
                    print(dec_tar[0:2], "\n\n")
            else:
                break
        if inference:
            prediction_ids = np.array(all_infer_ids, dtype=int)
            target_ids = np.array(all_target_ids, dtype=int)

            # average, greedy, extreme = embedding_eval(prediction_ids, target_ids, dataLoader.embedding_matrix)
            # print('Average {} | Greedy {} | Extreme {}\n\n'.format(average, greedy, extreme))

            # the_list = ["i don't know", "i have no idea", "i dont know", "I have no clue", "i'm not sure", "i am not sure"]
            # gen_count = 0
            # det_count = 0
            hyp = open('mains/hypothesis.txt', 'w')
            ref = open('mains/references.txt', 'w')
            for gen in range(len(all_detailed_response)):
                hyp_wri = all_detailed_response[gen] + "\n"
                hyp.write(hyp_wri)

                tru_wri = all_ground_truth[gen] + "\n"
                ref.write(tru_wri)
                # if any([check in all_general_response[gen] for check in the_list]):
                #     gen_count += 1
                # if not any([check in all_detailed_response[gen] for check in the_list]):
                #     det_count += 1
            # print("Out of", len(all_general_response), " general responses,", gen_count, "responses are general\nSo the proportion is: ", str(gen_count/len(all_general_response)))
            # print("Out of", len(all_general_response), " detailed responses,", det_count, "responses are detailed\nSo the proportion is: ", str(det_count/len(all_general_response)))
            hyp = open('mains/hypothesis.txt', 'r')
            ref = open('mains/references.txt', 'r')
            nlg_e = eval_emb_metrics(hyp, [ref])
            print(nlg_e)
            
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
            if last_improvement > 5:
                print('\n\nlast_improvement is: ', last_improvement)
                print("No improvements so cease training\n\n")
                break
                
            print('\n\nlast_improvement is: ', last_improvement)
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
                    X_prior = np.concatenate((X_prior, np.mean(prior_z, axis=0)))
                    X_post = np.concatenate((X_post, np.mean(post_z, axis=0)))
                    # model.train_decoder_session(sess, enc_inp, dec_inp, dec_tar, x_labels, y_labels)
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
            
            print("\nVisualize latent space: prior and posterior space")
            sns.scatterplot(
              x="pca-one", y="pca-two",
              hue="label",
              palette=sns.color_palette("hls", 2),
              data=df.loc[rndperm,:],
              legend="full",
              alpha=0.3
            )
            plt.show()

            print("\nLoss plot per iteration is: ")
            plt.plot(loss_list)
            plt.legend(['loss'], loc='upper right')
            plt.show()

            print("\nNegative log likelihood (nll) loss vs KL loss plot per iteration is: ")
            plt.plot(nll_loss_list)
            plt.plot(kl_loss_list)
            plt.legend(['nll loss', 'kl loss'], loc='upper right')
            plt.show()

            print("\nKL loss plot vs KL loss weight per iteration is: ")
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
            for (enc_inp, dec_inp, dec_tar, x_labels, y_labels, y_labels_general) in tqdm(dataLoader.test_generator(), desc="testing"):
                test_out = model.test_session(sess, enc_inp, dec_inp, dec_tar, x_labels, y_labels)
                test_loss += test_out['loss_test']
                test_nll_loss += test_out['nll_loss_test']
                test_kl_loss += test_out['kl_loss']
                test_count += 1
                test_loss_list = np.append(test_loss_list, test_out['loss_test'])
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
            print("\nTraining loss vs. Testing loss per epoch is: ")
            plt.plot(training_epoch_loss)
            plt.plot( validation_epoch_loss)
            plt.legend(['training loss', 'validation loss'], loc='upper right')
            plt.show()

            print()

            print('# sample test')
            self.sample_test(model, dataLoader, sess, inference=False)

            # Update model params using tensorflow operations
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



            