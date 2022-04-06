def file_list(file_name):
    """
    This function opens a file and returns it as a list.
    All new line characters are stripped.
    All lines that start with '#' are considered comments and are not included.
    :param file_name: the name of the file to be put into a list
    :return: a list containing each line of the file, except those that start with '#'
    """

    f_list = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            # if line[0] != '#' and line[0] != '\n' and len(line[0]) > 0:
            f_list.append(line.strip('\n'))
    return f_list


def convert(x):
    """
    convert 16 bit int x into two 8 bit ints, coarse and fine.
    """
    c = x >> 8  # The value of x shifted 8 bits to the right, creating coarse.
    f = x % 256  # The remainder of x / 256, creating fine.
    return c, f
