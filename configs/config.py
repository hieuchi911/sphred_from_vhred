args = {}
args['PAD_ID'] = 0
args['SOS_ID'] = 1
args['EOS_ID']=2
args['UNK_ID']=3

args['sphred-nucleus'] = False  # True -> improved SPHRED with Nucleus sampling, False -> original SPHRED in paper
args['rnn_type']='GRU'
args['keep_prob']=0.4
args['num_layer']=1
args['test_ratio']=0.1
args['num_pre_utterance']=3
args['learning_rate']=0.0001

args['batch_size']=64
args['n_epochs']=100
args['display_step']=50

args['vocab_size']=250000
args['num_sampled']=1000
args['word_dropout_rate']=0.3

args['max_len']=20
args['embed_dims']=100

args['rnn_size']=64
args['beam_width']=5
args['clip_norm']=5.0

args['vhred_ckpt_dir']='model/ckpt/squad-sphred-n'
args['vae_display_step']=100
args['latent_size']=64
args['anneal_max']=0.8
args['anneal_bias']=5000 # this is where (vanilla seq2seq, which aim at reconstruct only) model's training cost is identical to true ELBO
args['top_p'] = 0.15
args['top_k'] = 15

# args['discriminator_dropout_rate']=0.2
# args['n_filters']=128
# args['n_class']=2

# args['wake_sleep_display_step']=1
# args['temp_anneal_max']=1.0
# args['temp_anneal_bias']=1000
# args['lambda_c']=0.1
# args['lambda_z']=0.1
# args['lambda_u']=0.1
# args['beta']=0.1
