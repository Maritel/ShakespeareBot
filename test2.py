from hmmlearn.hmm import MultinomialHMM as MHMM
import dataprocessing

X, seq_lengths, conv_dict = dataprocessing.parse_words_lines()
print(X)
print(seq_lengths)
hmm = MHMM(n_components=5)

hmm.fit(X, seq_lengths)

generation = hmm.sample(10)

observations = generation[1]

translation = [conv_dict[i] for i in observations]
print(translation)


