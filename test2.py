import dataprocessing
import numpy as np
import copy
import HMM

# Label the state with the stress notation.

X, word_conv = dataprocessing.parse_words_lines()
word_lists = dataprocessing.get_word_lists()

stresses, syllables = dataprocessing.analyze_syllables(word_lists)
flattened_stresses = []
for i in range(len(stresses)):
    for j in range(len(stresses[i])):
        flattened_stresses.append(stresses[i][j])

unique_stresses, stress_counts = np.unique(flattened_stresses,
                                           return_counts=True)
print(unique_stresses)
print(stress_counts)
stress_dict = {}
for i in range(len(unique_stresses)):
    stress_dict[unique_stresses[i]] = i

Y = copy.deepcopy(stresses)
flattened_Y = []
for i in range(len(Y)):
    for j in range(len(Y[i])):
        flattened_Y.append(stress_dict[Y[i][j]])
        Y[i][j] = stress_dict[Y[i][j]]

unique_Ys, counts = np.unique(flattened_Y, return_counts=True)

hmm = HMM.unsupervised_HMM(X, n_states=len(unique_Ys), n_iters=0)

hmm.supervised_learning(X, Y)


for _ in range(5):
    emission = hmm.generate_emission(10)
    translation = [word_conv[i] for i in emission]
    print(translation)