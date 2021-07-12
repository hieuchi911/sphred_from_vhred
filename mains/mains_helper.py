import numpy as np
from data import ids_to_words
import warnings
# warnings.filterwarnings('error')

def embedding_eval(prediction_ids, target_ids, embedding, id_to_word):  # prediction_ids, weights, and target_ids: (batch_size, max_lentgh)
    pred = np.take(embedding, prediction_ids, axis=0) # (batch_size, max_lentgh, 100)
    targ = np.take(embedding, target_ids, axis=0)     # (batch_size, max_lentgh, 100)

    dim = np.shape(pred)[2]
    # Average:
    scores = []
    for i in range(len(prediction_ids)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        X = np.zeros((dim,))  # all prediction tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        Y = np.zeros((dim,))  # all target     tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if prediction_ids[i][j] not in [0, 1, 2, 3]:
                X += pred[i][j]
        X = np.array(X)/np.linalg.norm(X)
        
        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if target_ids[i][j] not in [0, 1, 2, 3]:
                Y += targ[i][j]
                
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                Y = np.array(Y)/np.linalg.norm(Y)
                o = np.dot(X, Y.T)/np.linalg.norm(X)/np.linalg.norm(Y)
            except Warning:
                print("Empty response")

        scores.append(o)
    scores = np.asarray(scores)
    average = np.mean(scores)

    # Greedy:
    g1 = asymetric_greedy(prediction_ids, target_ids, dim, pred, targ)
    g1 = np.mean(g1)
    g2 = asymetric_greedy(target_ids, prediction_ids, dim, targ, pred)
    g2 = np.mean(g2)
    greedy = (g1 + g2)/2.0


    # Extrema:
    scores = []
    for i in range(len(prediction_ids)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        X = []  # all prediction tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        Y = []  # all target     tokens embedding, shape is (seq_len, 100) where seq_len != max_len
        for j in range(len(prediction_ids[i])):   # for j'th token in the i'th batch of #max_len tokens
            if prediction_ids[i][j] not in [0, 1, 2, 3]:
                X.append(pred[i][j])
            if target_ids[i][j] not in [0, 1, 2, 3]:
                Y.append(targ[i][j])
        xmax = np.max(X, 0)  # get positive max
        xmin = np.min(X,0)  # get abs of min
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
            print("Empty response")

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

def asymetric_greedy(arr1, arr2, dim, a1_emb, a2_emb):
    scores = []
    for i in range(len(arr1)):  # Each batch is of size (max_len), this loop takes average on non-pad tokens (id=0)
        X = 0
        y_count = 0
        x_count = 0
        o = 0.0
        Y = 0
        flag = False
        for j in range(len(arr1[i])):   # for j'th token in the i'th batch of #max_len tokens
            if arr1[i][j] not in [0, 1, 2, 3]:
                if j == 0:
                    Y = a1_emb[i][j].reshape((dim,1))
                else:
                    # Y is a matrix whose columns are the word vectors, shape is (dim, seq_len)
                    try:
                        Y = np.hstack((Y,a1_emb[i][j].reshape((dim,1))))
                    except:
                        print("\n",Y)
                        print(a1_emb[i][j].reshape((dim,1)), "\n")                        
                # print(Y)
                y_count += 1
            else:
                if j == 0:
                    flag = True
                    break
        if flag:
            continue
        for j in range(len(arr1[i])):   # for j'th token of sequence arr2 in batch j
            if arr2[i][j] not in [0, 1, 2, 3]:                
                emb = a2_emb[i][j].reshape((1,dim))
                tmp  = emb.dot(Y)/np.linalg.norm(emb)/np.linalg.norm(Y, axis=0)
                o += np.max(tmp)       # take out the max result
                x_count += 1
            else:
                if j == 0:
                    break

        # if none of the words in response or ground truth have embeddings, count result as zero
        if x_count < 1 or y_count < 1:
            scores.append(0)
            continue

        o /= float(x_count)
        scores.append(o)
    return np.asarray(scores)

if __name__ == "__main__":
    p = np.array([[0, 2, 5, 6, 0, 0],#])
        [4, 3, 5, 0, 0, 0],
        [3, 2, 1, 6, 0, 0],
        [5, 4, 0, 0, 0, 0],
        [1, 2, 5, 6, 2, 1]])

    t = np.array([[1, 1, 4, 6, 2, 0],
        [2, 6, 5, 5, 0, 0],
        [3, 4, 6, 4, 0, 0],
        [4, 3, 0, 0, 0, 0],
        [5, 2, 1, 3, 2, 1]])

    em = np.array([[0.1, 0.2, 0.14], [-0.005, -0.3, 0.19], [0.2, -0.2, 0.2],
        [0.03, 0.6, -0.6], [0.01, -0.22, 0.5], [0.3, 0.2, -0.95], [0.2, 0.3, 0.05]])

    p = np.random.randint(6, size=(60000, 6))
    # print(p)
    # print(pm)
    # print(p)

    res = embedding_eval(p, p, em*5)
    print(res)