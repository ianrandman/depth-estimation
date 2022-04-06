#!/usr/bin/python

from PIL import Image
import numpy as np
from scipy.interpolate import NearestNDInterpolator

from utils.interpolate_background import lin_interp


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    # lin_interp(depth.shape, depth)
    depth[depth_png == 0] = -1.

    # mask = npzrp(*np.indices(depth.shape))

    return depth
