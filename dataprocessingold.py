import numpy as np
import re


def parse_lines(filename='data/shakespeare.txt',
                ignore_chars='[\n,;:.!?()]',
                padding=True):
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
    sequences = []

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

    sequences, conversion_list = items_to_numbers(word_lists)

    if padding:
        sequences = pad(sequences)

    return sequences, conversion_list
    pass


def items_to_numbers(item_lists):
    """
    Replaces a list of sequences of items with integer observations.
    :param item_lists: A list of lists of strings (words)
    :return: sequences: A list of emission sequences
    :return: conversion_list: A list, with ith element being the item for
        integer i
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
    sequences = item_lists
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