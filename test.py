import poetrytools
import dataprocessing
from nltk import pos_tag
import nltk

raw = "Hi there. My name is Professor Oak."
poem = ['No', 'more', 'swords', 'no', 'more', 'dreams']

token = nltk.word_tokenize(raw)
print(token)
print(pos_tag(token))
print(pos_tag(poem))