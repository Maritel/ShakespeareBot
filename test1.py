import dataprocessing
import numpy as np
import HMM

X, conversion_list = dataprocessing.parse_words_lines()

param_n_states = [5, 10, 20, 40, 80]
lines = []
emissions = []
for n_states in param_n_states:
    n_observations = len(conversion_list)

    hmm = HMM.unsupervised_HMM(X, n_states, n_iters=2)
    emission = hmm.generate_emission(M=10)

    # i-1 because the observations are 1-indexed
    translated_emission = [conversion_list[i] for i in emission]
    line = ''
    for word in translated_emission:
        line += word + ' '
    lines.append(line[0:len(line) - 1])
    emissions.append(emission)

print(lines)
print(emissions)