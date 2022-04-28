"""
Utilities relating to files.

__author__ = Ian Randman
"""
import numpy as np
from PIL import Image


def file_list(file_name):
    """
    This function opens a file and returns it as a list.
    All new line characters are stripped.
    :param file_name: the name of the file to be put into a list
    :return: a list containing each line of the file
    """

    f_list = []
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            f_list.append(line.strip('\n'))
    return f_list


def convert(x):
    """
    convert 16 bit int x into two 8 bit ints, coarse and fine.
    """
    c = x >> 8  # The value of x shifted 8 bits to the right, creating coarse.
    f = x % 256  # The remainder of x / 256, creating fine.
    return c, f


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    depth[depth_png == 0] = -1.

    return depth