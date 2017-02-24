# Creation of rhyme dictionary

import dataprocessing
import numpy as np
import poetrytools
import copy
import HMM
import pickle

# Label the state with the stress notation.

hmm_file = '15states100iterations.txt'
with open(hmm_file, 'rb') as file:
    hmm = pickle.load(file)

print(np.array(hmm.A))

X, word_conv = dataprocessing.parse_words_lines()
word_lists = dataprocessing.get_word_lists()

lines = []
for _ in range(14):
    emission = hmm.generate_emission(8)
    emission.reverse()
    lines.append(emission)

for line in lines:
    for num in line:
        print(str(word_conv[num]), end=' ')
    print()




