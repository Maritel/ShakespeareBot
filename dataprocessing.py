import numpy as np
import re
import poetrytools
import copy
import nltk

def parse_words_lines(filename='data/shakespeare.txt',
                      ignore_chars="[\n,';:.!?()]"):
    """
    Parse raw text into observation sequences, assuming that one line is a
    sequence and one word is an observation. Splitting was done by interpreting
    a space as a delimiter.

    :param filename: filename to read from
    :param ignore_chars: regex to ignore specific characters in each line
    :param padding: whether to pad sequences to make them equal length
    :return: sequences: a list of integer observation sequences
    :return: conversion_list: a list whose i'th element is the item
        corresponding to observation i
    """

    word_lists = get_word_lists(filename=filename, ignore_chars=ignore_chars)

    sequences, conversion_list = items_to_numbers(word_lists)

    return sequences, conversion_list
    pass


def get_word_lists(filename='data/shakespeare.txt',
                   ignore_chars="[\n,';:.!?()]"):
    """
    Translates raw text into a list of lists of words.
    :param filename: file to read form
    :param ignore_chars: characters to ignore completely
    :return: word_lists: list of lists of word. Each list is a line.
    """
    with open(filename) as file:
        content = file.readlines()

        # Interpret line in lower case only.
        # Strip punctuation and end-of-line characters.
        content = [line.lower() for line in content]
        content = [re.sub(pattern=ignore_chars, repl='', string=line)
                   for line in content]

        word_lists = [line.split() for line in content]

        # Lines of length below 2 are not even text.
        word_lists = [words for words in word_lists if len(words) >= 2]

    return word_lists


def analyze_syllables(word_lists):
    """
    For each word in the list of lists of words, return its syllable stresses
    and its syllable count.
    :param word_lists: A list of lists of word.
    :return: stresses: Stress notations for each word. Same shape as
    'word_lists'
    :return: syllables: Syllable counts for each word. Same shape as
    'word_lists'
    """
    stresses = poetrytools.scanscion(word_lists)

    # deep copy
    syllables = copy.deepcopy(stresses)
    for i in range(len(syllables)):
        for j in range(len(syllables[i])):
            syllables[i][j] = len(syllables[i][j])

    return stresses, syllables


def analyze_POS(word_lists):
    """
    Return the POS of each word.
    :param word_lists: A list of lists of words
    :return: POS: parts of speech for every word. Same shape as 'word_lists'
    """
    tagged = copy.deepcopy(word_lists)

    for i in range(len(tagged)):
        print(i)
        tagged[i] = nltk.pos_tag(tagged[i])

    for i in range(len(tagged)):
        for j in range(len(tagged[i])):
            print(str(i) + ',' + str(j))
            tagged[i][j] = tagged[i][j][1]

    return tagged


def items_to_numbers(item_lists):
    """
    Replaces a list of sequences of items with integers.
    :param item_lists: A list of lists of strings (words)
    :return: sequences: A list of lists of integers
    :return: conversion_list: A list, with ith element being the item
        corresponding to integer i
    """

    # Flatten the array
    flattened_items = []
    for item_list in item_lists:
        for item in item_list:
            flattened_items.append(item)

    unique_items = np.unique(flattened_items)

    # Create a dictionary to convert items
    item_dict = {}
    for i in range(len(unique_items)):
        item_dict[unique_items[i]] = i

    # Convert all items
    sequences = copy.deepcopy(item_lists)
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            sequences[i][j] = item_dict[sequences[i][j]]

    return sequences, unique_items

def pad(sequences, pad_value=-1):
    """
    Convert sequences into a 2D matrix. 'pad_value' is used to signal that the
    line has already ended.

    :param sequences: A list of integer observation sequences
    :param pad_value: The value to pad with.
    :return: padded_sequences:
    """

    # Find the maximum-length sequence.
    max_length = -float('inf')
    for sequence in sequences:
        if len(sequence) > max_length:
            max_length = len(sequence)

    # Initialize the 2D array with -1 in every entry.
    padded_sequences = np.full(shape=[len(sequences), max_length],
                               fill_value=-1,
                               dtype=np.int)

    # Fill the values that are known.
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            padded_sequences[i][j] = sequences[i][j]

    return padded_sequences


if __name__ == "__main__":
    pass