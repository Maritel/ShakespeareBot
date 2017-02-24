import dataprocessing
import numpy as np
import HMM
import pickle
import copy

X_raw, conversion_list = dataprocessing.parse_words_lines()

print(X_raw)
# REVERSE X
X = copy.deepcopy(X_raw)
for i in range(len(X_raw)):
    X[i].reverse()
print(X)

n_states = 50
lines = []
emissions = []
n_observations = len(conversion_list)

hmm = HMM.unsupervised_HMM(X, n_states, n_iters=10)

for _ in range(10):
    emission = hmm.generate_emission(M=10)

    # i-1 because the observations are 1-indexed
    translated_emission = [conversion_list[i] for i in emission]

    line = ''
    for word in translated_emission:
        line += word + ' '
    lines.append(line[0:len(line) - 1])
    emissions.append(emission)
    print(line)

with open('50states10iterations.txt', 'wb') as file:
    pickle.dump(hmm, file)