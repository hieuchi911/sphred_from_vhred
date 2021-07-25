import numpy as np
from data import ids_to_words
import warnings
from gensim.models import Word2Vec

# Embedding based evaluation metrics: Average Embedding, Greedy Matching, and Vector Extrema

def embedding_eval(prediction_ids, target_ids, embedding):  # prediction_ids, weights, and target_ids: (batch_size, max_lentgh)
    pred = np.take(embedding, prediction_ids, axis=0) # (batch_size, max_lentgh, 100)
    targ = np.take(embedding, target_ids, axis=0)     # (batch_size, max_lentgh, 100)

    dim = np.shape(pred)[2]
    # Average:
    scores = []
    for i in range(len(prediction_ids)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        X = np.zeros((dim,))  # all prediction tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        Y = np.zeros((dim,))  # all target     tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        if max(prediction_ids[i]) < 4 or max(target_ids[i]) < 4:
            scores.append(0.0)
            continue
        count_x = 0
        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if prediction_ids[i][j] not in [0, 1, 2, 3]:
                X += pred[i][j]
                count_x += 1
            else:
                if prediction_ids[i][j] == 2:
                    break
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                X = X/count_x
                X = np.array(X)/np.linalg.norm(X)
            except:
                print("AVERAGE\tEmpty predicted response")
                scores.append(0.0)
                continue  
        count_y = 0    
        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if target_ids[i][j] not in [0, 1, 2, 3]:
                Y += targ[i][j]
                count_y += 1
            else:
                if target_ids[i][j] == 2:
                    break
                
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                Y = Y/count_y
                Y = np.array(Y)/np.linalg.norm(Y)
                o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
            except Warning:
                print("AVERAGE\tEmpty target")
                scores.append(0.0)
                continue

        scores.append(o)
    scores = np.asarray(scores)
    average = np.mean(scores)

    # Greedy:
    g1 = asymetric_greedy(prediction_ids, target_ids, dim, pred, targ)
    print(g1)
    g1 = np.mean(g1)
    print(g1)
    g2 = asymetric_greedy(target_ids, prediction_ids, dim, targ, pred, a1_p=False)
    print(g2)
    g2 = np.mean(g2)
    print(g2)
    greedy = (g1 + g2)/2.0


    # Extrema:
    scores = []
    for i in range(len(prediction_ids)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        X = []  # all prediction tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        Y = []  # all target     tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        if max(prediction_ids[i]) < 4 or max(target_ids[i]) < 4:
            scores.append(0.0)
            continue

        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if prediction_ids[i][j] not in [0, 1, 2, 3]:
                X.append(pred[i][j])
            else:
                if prediction_ids[i][j] == 2:
                    break
        for j in range(len(target_ids[i])):
            if target_ids[i][j] not in [0, 1, 2, 3]:
                Y.append(targ[i][j])
            else:
                if target_ids[i][j] == 2:
                    break
        try:
            xmax = np.max(X, 0)  # get positive max
            xmin = np.min(X,0)  # get abs of min
        except:
            print("EXTREMA\tEmpty predicted response")
            scores.append(0.0)
            continue
        xtrema = []
        for a in range(len(xmax)):
            if np.abs(xmin[a]) > xmax[a]:
                xtrema.append(xmin[a])
            else:
                xtrema.append(xmax[a])
        X = np.array(xtrema)   # get extrema
        
        try:
            ymax = np.max(Y, 0)
            ymin = np.min(Y,0)
        except:
            print("EXTREMA\tEmpty target")
            scores.append(0.0)
            continue

        ytrema = []
        for i in range(len(ymax)):
            if np.abs(ymin[i]) > ymax[i]:
                ytrema.append(ymin[i])
            else:
                ytrema.append(ymax[i])
        Y = np.array(ytrema)

        o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)

        scores.append(o)
    scores = np.asarray(scores)
    extrema = np.mean(scores)

    return average, greedy, extrema

def asymetric_greedy(arr1, arr2, dim, a1_emb, a2_emb, a1_p=True):
    scores = []
    for i in range(len(arr1)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        if max(arr1[i]) < 4 or max(arr2[i]) < 4:
            scores.append(0.0)
            continue
            
        y_count = 0
        x_count = 0
        o = 0.0
        Y = 0
        flag = False
        for j in range(len(arr1[i])):   # for j'th token in the i'th batch of #max_len tokens
            if arr1[i][j] not in [0, 1, 2, 3]:
                if str(type(Y)) != "<class 'numpy.ndarray'>":
                    Y = a1_emb[i][j].reshape((dim,1))
                else:
                    # Y is a matrix whose columns are the word vectors, shape is (dim, seq_len)
                    try:
                        Y = np.hstack((Y,a1_emb[i][j].reshape((dim,1))))
                    except:
                        print("\n",Y)
                        print(a1_emb[i][j].reshape((dim,1)), "\n")
                y_count += 1
            else:
                if arr1[i][j] == 2:
                    if str(type(Y)) != "<class 'numpy.ndarray'>":
                        flag = True
                    break
                else:
                    flag = False
        if flag:
            scores.append(0.0)
            continue
        emb = 0
        for j in range(len(arr1[i])):   # for j'th token of sequence arr2 in batch j
            if arr2[i][j] not in [0, 1, 2, 3]:          
                emb = a2_emb[i][j].reshape((1,dim))
                emb = np.array(emb)/np.linalg.norm(emb)
                Y = np.array(Y)/np.linalg.norm(Y, axis=0)
                
                tmp = emb.dot(Y)/np.linalg.norm(emb)/np.linalg.norm(Y, axis=0)
                o += np.max(tmp)       # take out the max result
                x_count += 1
            else:
                if arr2[i][j] == 2:
                    if str(type(emb)) != "<class 'numpy.ndarray'>":
                        flag = True
                    break
                else:
                    flag = False
        if flag:
            scores.append(0.0)
            continue
        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0.0)
            continue

        o /= float(x_count)
        scores.append(o)
    return np.asarray(scores)

if __name__ == "__main__":
    p = np.array([[1186, 4, 84, 278, 61, 61, 154, 21, 18, 34, 150, 459, 4, 214, 278],
        [561,   4,  84, 278, 154,  21, 150,  21, 150,  27, 189, 225,   4, 214, 278],
        [1186,  21, 33, 278, 34, 1240, 18, 713, 189, 64,  164, 4, 214, 278, 278],
        [  4,  95,  21,  21,  21,  22,  21,  95,  21,  22,  95,  95, 170,  95,  21],
        [444,   4, 214,  53, 215,   4, 154,  18, 643,  38, 160,  41,   4, 214, 278]])

    t = np.array([[447, 38, 29, 459, 2, 0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [  1129,    120,   1692,    366,     64, 198307,    352, 198308,   1170,    646,
   2553,    646,   2187,    475,      2],
        [  59,  643,    9, 7426,  859,   15,   57, 2499,  405,   59,  643,    9, 7426,  859,
    2],
        [ 371,  512,   95, 1648,   95,   60,   38, 44,  734,    2,    0,    0,    0,    0,
    0],
        [76313,  2399,    22,    21,    70,  1503,    84,    73,   312,     2,     0,     0,
     0,     0,     0]])
    
    new_model = Word2Vec.load("data/Ubuntu_model.bin")
    embedding_matrix = []
    for i in range(4):
      embedding_matrix.append(np.zeros(100))
    p = ["hello this is is the", "well this this the linux", "where where it it it",
         "to to to just this", "what what linux linux linux"]
    # p = ["you should do this now", "people reported some great news", "i should have been there",
    #      "it is made perfectly perfect", "aid will be provided now"]
    # p = ["only one thing to do", "the record received great new", "i should have been there",
    #      "it is made perfectly perfect", "aid will be provided now"]
    t = ["all you need to do", "great news for the record", "i wish i was here",
         "this is so carefully done", "hey you need help right"]
    words = new_model.wv.index_to_key
    for word in words:
      if '<pad>' in word:
        break
      vector = new_model.wv[word]
      embedding_matrix.append(vector)
    output_rt = np.array(embedding_matrix)