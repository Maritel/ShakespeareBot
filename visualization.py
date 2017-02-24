import dataprocessing
import numpy as np
import HMM
import nltk
import poetrytools

X, conv_words = dataprocessing.parse_words_lines()

n_states = 5
n_observations = len(conv_words)

hmm = HMM.unsupervised_HMM(X, n_states, n_iters=300)
emission = hmm.generate_emission(M=10)

# i-1 because the observations are 1-indexed
translated_emission = [conv_words[i] for i in emission]
line = ''
for word in translated_emission:
    line += word + ' '
print(line)
print(emission)

# Print transition matrix.
A_vis = np.around(np.array(hmm.A), 4)
print(A_vis)

# Obtain the top 10 words for each state.
O = np.array(hmm.O)
for i in range(len(O)):
    state_emissions = O[i, :]
    top_emissions = state_emissions.argsort()[-10:][::-1]
    top_probs = [state_emissions[j] for j in top_emissions]
    top_words = [conv_words[j] for j in top_emissions]

    pos = nltk.pos_tag(top_words)
    pos = [x[1] for x in pos]

    stresses = poetrytools.scanscion([top_words])[0]

    print()
    print(top_words)
    print(np.around(top_probs, 4))
    print(pos)
    print(stresses)