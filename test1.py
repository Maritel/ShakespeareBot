import dataprocessingold
import numpy as np
import HMM

X, conversion_list = dataprocessingold.parse_lines()

n_states = 10 # hyperparameter
n_observations = len(conversion_list)

hmm = HMM.unsupervised_HMM(X, n_states, n_iters=10)
emission = hmm.generate_emission(M=10)

# i-1 because the observations are 1-indexed
translated_emission = [conversion_list[i-1] for i in emission]
line = ''
for word in translated_emission:
    line += word + ' '
print(line[0:len(line) - 1])

zealous_index = 1 + conversion_list.tolist().index('zealous')
print(np.array(hmm.O)[:, zealous_index])
print(len(conversion_list))
