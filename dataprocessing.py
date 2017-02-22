import numpy as np


def parse_lines(filename='data/shakespeare.txt'):
    """
    Parse raw text into observation sequences, assuming that one line is a
    sequence and one word is an observation. Splitting was done by interpreting
    a space as a delimiter.

    All punctuation, except apostrophes, is stripped. All words are converted
    to lowercase.

    :param filename: Name of file to read data from
    :return: A list of lists (sequences)
    """
    sequences = []

    with open(filename) as file:
        content = file.readlines()

        # Interpret line in lower case only.
        # Strip punctuation and end-of-line characters.
        content = [line.lower().strip('[\n,;:.!?]()') for line in content]

        word_lists = [line.split() for line in content]

        # Lines of length below 2 are not even text.
        word_lists = [words for words in word_lists if len(words) >= 2]

    sequences, conversion_dict = items_to_numbers(word_lists)

    return sequences, conversion_dict
    pass


def items_to_numbers(item_lists):
    """
    Replaces a list of sequences of
    :param item_lists: A list of lists of strings (words)
    :return:
        sequences: A list of emission sequences
        conversion_list: A list, with ith element being the item for integer i
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

if __name__ == "__main__":
    parse_result = parse_lines()
    print(parse_result[0])
    print(parse_result[1])