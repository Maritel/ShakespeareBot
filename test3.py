import dataprocessing
import numpy as np
import pickle
import copy
import HMM

# Label the state with the part of speech.

X, word_conv = dataprocessing.parse_words_lines()
word_lists = dataprocessing.get_word_lists()
print(X)
for i in range(len(X)):
    X[i].reverse()
print(X)
print(word_lists)
for i in range(len(word_lists)):
    word_lists[i].reverse()
print(word_lists)

# Load up saved pos
with open('pos.txt', 'rb') as file:
    pos = pickle.load(file)

flattened_pos = []
for i in range(len(pos)):
    for j in range(len(pos[i])):
        flattened_pos.append(pos[i][j])

unique_pos, counts = np.unique(flattened_pos, return_counts=True)
pos_dict = {}
for i in range(len(unique_pos)):
    pos_dict[unique_pos[i]] = i

Y = copy.deepcopy(pos)
flattened_Y = []
for i in range(len(Y)):
    for j in range(len(Y[i])):
        flattened_Y.append(pos_dict[Y[i][j]])
        Y[i][j] = pos_dict[Y[i][j]]
unique_Ys = np.unique(flattened_Y)


hmm = HMM.unsupervised_HMM(X, n_states=len(unique_Ys), n_iters=0)

hmm.supervised_learning(X, Y)

with open('poshmm.txt', 'wb') as file:
    pickle.dump(hmm, file)
