import numpy as np
import re

def parse_words_lines(filename='data/shakespeare.txt',
                      ignore_chars='[\n,;:.!?()]',
                      min_line_length=3):
    """
    Takes a text file and converts it to a list of samples.
    Each word is a sample, each line is a sequence.

    :param filename: filename of data to read
    :param ignore_chars: regexp of characters to disregard
    :param min_line_length: Lines with less than this number of words are
    disregarded.
    :return: samples: shape(n_samples,1), an effectively 1D array of samples.
    Note that samples are represented by nonnegative integers.
    :return: lengths: shape(n_sequences,) a 1D array of the lengths of the
    individual sequences. The sum of the lengths should be n_samples.
    :return: conv_list: shape(n_observations,) a 1D array whose i'th element is
    the word corresponding to observation i
    """

    # First, get a list of lists of words.
    word_lists = []
    with open(filename) as file:
        content = file.readlines()

        # Interpret line in lower case. Strip the ignored characters.
        content = [line.lower() for line in content]
        content = [re.sub(pattern=ignore_chars, repl='', string=line)
                   for line in content]

        word_lists = [line.split() for line in content]

        # Disregard lines that are too short.
        word_lists = [word_list for word_list in word_lists
                      if len(word_list) >= min_line_length]

    sequences, conv_list = items_to_numbers(word_lists)

    # Flatten 'sequences' into a list of samples, but retain sequence lengths.
    lengths = np.array([len(sequence) for sequence in sequences])
    samples = []
    for sequence in sequences:
        for sample in sequence:
            samples.append(sample)
    samples = np.array(samples)
    samples = np.reshape(a=samples, newshape=(len(samples), 1))

    return samples, lengths, conv_list


def items_to_numbers(item_lists):
    """
    Replaces a list of sequences of items with integer observations.
    :param item_lists: A list of lists of strings (words)
    :return: sequences: A list of integer-observation sequences
    :return: conv_list: A list, with ith element being the item corresponding
        to observatoin i
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
