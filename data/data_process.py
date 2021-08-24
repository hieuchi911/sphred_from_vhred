import nltk
import os
import tensorflow as tf
from configs import args
import random
import numpy as np
from gensim.models import Word2Vec

def word_dropout(x):
    is_dropped = np.random.binomial(1, args['word_dropout_rate'], np.shape(x))
    fn = np.vectorize(lambda x, k: args['UNK_ID'] if (
        k and (x not in range(4))) else x)
    return fn(x, is_dropped)

def word_embeddings(words): # word list here contains special tokens
    curPath = os.path.abspath(os.path.dirname(__file__))
    # file = os.path.join(curPath, 'Ubuntu_model.bin')
    file = os.path.join(curPath, 'Squad_model.bin')
    new_model = Word2Vec.load(file)
    embedding_matrix = []
    for i in range(4):
      embedding_matrix.append(np.zeros(100))
    for word in words:
      if '<pad>' in word:
        break
      vector = new_model.wv[word]
      embedding_matrix.append(vector)
    output_rt = np.array(embedding_matrix)
    print("Shape of output_rt", np.shape(output_rt))
    return output_rt

def read_data(filename=None, utterances=None, flag=None):
    ret = []
    ret_label = []
    if filename:    # This is for training
        general = ["i don't know", "i have no idea", "i dont know", "I have no clue", "i'm not sure", "i am not sure"]
        curPath = os.path.abspath(os.path.dirname(__file__))
        file = os.path.join(curPath, filename)
        with open(file, "r", encoding='utf-8') as f:
            # UNCOMMENT THIS TO MAKE TOY DATASET
            # count = 0
            for line in f:
                # UNCOMMENT THIS TO MAKE TOY DATASET
                # This upper limit (call it U) must be greater than 4*batch_size and round_down(U*(1-test_ratio)/batch_size) >= display_step, modify U or display_step
                # if count > 500:
                #     break
                line = line.replace("\n", "")
                sents = line.split(" __eot__ ")
                conv = []
                labels = []
                for i in range(len(sents)):
                    if i > 3:
                        break
                    sent = sents[i]
                    sent = sent.lower().strip() # lower and strip off spaces in the string
                    words = nltk.word_tokenize(sent)
                    conv.append(words)
                    # CVAE: Flags
                    if general[0] in sent or general[1] in sent or general[2] in sent:
                        labels.append([0.0, 1.0])
                    else:
                        labels.append([1.0, 0.0])
                if len(conv) < 4:
                    continue

                ret.append(conv)
                ret_label.append(labels)
                
                # UNCOMMENT THIS TO MAKE TOY DATASET
                # count += 1
        
        return ret, ret_label
    else:        # This is for testing
        line = utterances.replace("\n", "")
        sents = line.split(" __eot__ ") # (3)
        conv = []
        labels = []
        for i in range(len(sents)):
            sent = sents[i]
            sent = sent.lower().strip() # lower and strip off spaces in the string
            words = nltk.word_tokenize(sent)  # (sentence_length)
            conv.append(words)    # (3, sentenence_length)
            # CVAE: Flags
            labels.append(flag)   # (3, 2)

        ret.append(conv)  # (1, 3, sentence_length)
        ret_label.append(labels)  # (1, 3, 2)

        return ret, ret_label

def read_vocab(filename):
    word_list = []
    curPath = os.path.abspath(os.path.dirname(__file__))
    file = os.path.join(curPath, filename)
    with open(file, "r", encoding='utf-8') as f:
        for line in f:
            line = line.replace("\n", "").lower().strip()
            word_list.append(line)

    word_to_id = {}
    id_to_word = {}
    i = 0
    for w in word_list:
        if w not in word_to_id.keys():
          word_to_id[w] = i + 4
          id_to_word[i + 4] = w
          i += 1

    word_to_id["<pad>"] = 0
    word_to_id["<sos>"] = 1
    word_to_id["<eos>"] = 2
    word_to_id["<unk>"] = 3

    id_to_word[-1] = "-1"
    id_to_word[0] = "<pad>"
    id_to_word[1] = "<sos>"
    id_to_word[2] = "<eos>"
    id_to_word[3] = "<unk>"

    # print(list(word_to_id.items())[:10])
    # print(list(word_to_id.items())[50:60])

    return word_to_id, id_to_word

def tokenize_data(data, word_to_id):  # data here is the file that contains all dialogs, so it has punctuations
    unk_id = word_to_id["<unk>"]

    ret = []
    for conv in data:
        padded_conv = []
        for turn in conv:
            words = [word_to_id.get(w, unk_id) for w in turn] # dict.get(a, b) return b if a is not in the dict
            words = words[:min(args['max_len']-1, len(words))]
            padded_conv.append(words)
        ret.append(padded_conv)
        
    return ret

def ids_to_words(example, id_to_word, is_pre_utterance=True):
    result = []
    if is_pre_utterance:
        for utterance in example:
            sentence = ' '.join([id_to_word.get(id) for id in utterance if id > 0])
            result.append(sentence)
        return '-->'.join(result)
    else:
        the_text = ''
        for id_ in example:
          if id_ > 0:
            the_text += id_to_word.get(id_) + " "
        return the_text

def shuffle_in_unison(a, b):
    a = np.array(a)
    b = np.array(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def split_data(data, labels):
    test_ratio = args['test_ratio']
    num_all_examples = len(data)
    num_test = int(num_all_examples * test_ratio)
    data, labels = shuffle_in_unison(data, labels)
    
    test_data = data[:num_test]
    train_data = data[num_test:]
    
    label_data = labels[num_test:]
    label_test_data = labels[:num_test]
    
    X_train = []
    x_labels = []
    
    y_train = []
    y_labels = []
    
    X_test = []
    x_test_labels = []

    y_test = []
    y_test_labels = []
    y_test_labels_general = []

    for i in range(len(train_data)): # dialogue in train_data:
        X_train.append(train_data[i][:3]) # [[first utterance], [second utterance], [third utterance]]
        x_labels.append(label_data[i][:3])

        y_train.append(train_data[i][3]) # [fourth utterance]
        y_labels.append(label_data[i][3])
    
    for i in range(len(test_data)):
        X_test.append(test_data[i][:3]) # [[first utterance], [second utterance], [third utterance]]
        x_test_labels.append(label_test_data[i][:3])
        
        y_test.append(test_data[i][3]) # [fourth utterance]
        y_test_labels.append([1.0, 0.0])
        y_test_labels_general.append([0.0, 1.0])

    return X_train, y_train, X_test, y_test, x_labels, y_labels, x_test_labels, y_test_labels, y_test_labels_general

def form_input_data(X=None, y=None, realtime_test=False, dialogue=None):
    enc_inp = []
    dec_inp = []
    dec_out = []
    if realtime_test:
        utterance_list = []
        for utterance in dialogue[0]: # The first, which is the only too, dialogue
            utterance_list.append(np.array(utterance + [args['EOS_ID']] + [args['PAD_ID']] * (args['max_len'] - 1 - len(utterance))))
        enc_inp.append(np.array(utterance_list))
        return enc_inp
    for dialogue in X:
        utterance_list = []
        for utterance in dialogue:
            to_append = np.array(utterance + [args['EOS_ID']] + [args['PAD_ID']] * (args['max_len'] - 1 - len(utterance)))
            utterance_list.append(to_append)
            # if len(to_append) != 15:
            #     print("weird utterance len:", to_append)
            #     print("\t", utterance)
        enc_inp.append(np.array(utterance_list))
    for dialogue in y:
        dec_inp.append(np.array([args['SOS_ID']] + dialogue + [args['PAD_ID']] * (args['max_len'] - 1  - len(dialogue))))
        dec_out.append(np.array(dialogue + [args['EOS_ID']] + [args['PAD_ID']] * (args['max_len'] - 1  - len(dialogue))))
    # print("Enc inp: ", enc_inp)
    print(enc_inp[0])
    enc_inp = np.array(enc_inp)
    dec_inp = word_dropout(np.array(dec_inp))
    dec_out = np.array(dec_out)

    return enc_inp, dec_inp, dec_out


