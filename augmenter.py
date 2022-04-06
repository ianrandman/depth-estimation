from math import floor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
import cv2
from skimage import measure
from skimage.measure import label, regionprops, regionprops_table


from utils.file_utils import convert
from utils.labels import id2label

# id_to_color_func = np.vectorize(lambda id: id2label[id].color)


cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)


def is_valid(depth):
    return depth > 0


def instance_segmentation_augment(instance_map, depth_map):
    unique_segs = np.unique(instance_map)
    assert len(cmaplist) > len(unique_segs)  # TODO case where too many instances

    # Pick the colors that instance segmentations will be assigned to
    # Do so as evenly as possible for increased contrast
    step = int(floor(len(cmaplist) / len(unique_segs)))
    last = len(unique_segs) * step  # this prevents us from getting M+1 elements
    indices = np.array([i for i in range(0, last, step)])
    np.random.shuffle(indices)

    # assign the instance segmentations to colors
    instance_to_color = dict()
    for seg_id, color_idx in zip(unique_segs, indices):
        instance_to_color[seg_id] = cmaplist[color_idx]

    # create an RGB image for the instance segmentation
    id_to_color_func = np.vectorize(lambda id: instance_to_color[id])
    instance_rgb = np.dstack(id_to_color_func(instance_map))

    # change the depth map

    # # semantic_seg, instance_ids = np.apply_along_axis(convert, 0, instance_map)
    # num_objects = 0
    # for seg_id in unique_segs:
    #     pixel_ids = np.where(instance_map == seg_id)
    #     mask = np.ma.masked_where(instance_map == seg_id, instance_map)
    #     mask = mask.mask.astype(int)
    #     num_objects += measure.label(mask).max()
    #     assert measure.label(mask).max() > 0
    #     # instance_rgb[pixel_ids] = (0, 0, 0, 0)
    #
    #     regions = regionprops(mask)
    #
    #     x=1

    # some instances in the instance map may be disconnected, so we want to
    # isolate each region of each instance
    label_map = measure.label(instance_map)
    # new_depth_map = depth_map.copy()
    for label in np.unique(label_map):
        # if not label == 62:
        #     continue

        pixel_ids = np.where((label_map == label) & (is_valid(depth_map)))
        # pixel_ids = np.where(label_map == label)
        # depth_map[pixel_ids] = np.mean(depth_map[pixel_ids][depth_map[pixel_ids] != -1])
        depth_map[pixel_ids] = np.mean(depth_map[pixel_ids])
        # depth_map[pixel_ids] = 1000
        x = 1
    # plt.imshow(instance_rgb)
    # plt.show()
    # x=1

    return instance_rgb, depth_map, label_map
