"""
The main functions for performing data augmentations.
The augmentations are instance segmentation, vertical position, and apparent size.

__author__ = Ian Randman
"""

from math import floor

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from PIL import Image


from utils.file_utils import convert
from utils.labels import id2label, category2categoryId


NUM_OBJECTS_TO_AUGMENT = 5
CATEGORIES_TO_AUGMENT = [category2categoryId[category] for category in ['vehicle', 'object', 'human']]


cmap = plt.cm.jet  # define the colormap
# extract all colors from the .jet map
cmaplist = [cmap(i) for i in range(cmap.N)]
# force the first color entry to be grey
cmaplist[0] = (.5, .5, .5, 1.0)


def is_valid(depth):
    return depth > 0


def identify_objects_to_augment(instance_map):
    # only "significant" objects within the scene should be considered candidates for augmenting

    # separate the instance segmentation map into semantic labels and instance labels
    semantic_map, instance_id_map = np.apply_along_axis(convert, 0, instance_map)

    # create a mask of pixels that are only part of valid categories
    semantic_to_categoryId_func = np.vectorize(lambda semantic_id: id2label[semantic_id].categoryId)
    categoryId_map = semantic_to_categoryId_func(semantic_map)
    mask = np.where(np.isin(categoryId_map, CATEGORIES_TO_AUGMENT), instance_map, 0)

    # get the object IDs from the mask of valid categories
    object_ids, counts = np.unique(mask, return_counts=True)
    object_ids = object_ids[1:]  # do not care about background
    counts = counts[1:]  # number of pixels for each object

    # sort objects by size in descending order
    indices = np.argsort(counts)[::-1]
    object_ids = object_ids[indices]

    # return object IDs sorted in descending order of the size of the object
    # also return a mask for the candidate objects
    return object_ids, mask


def instance_segmentation_augment(instance_map, depth_map):
    # perform the instance segmentation augmentation
    # the new image is a colorized version of the instance segmentation map
    # the new depth map replaces depth within an object region as the average of the depths within that region

    # get all instance IDs
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
    instance_rgb = np.array(Image.fromarray((instance_rgb * 255).astype(np.uint8)).convert('RGB'))

    # change the depth map

    # some instances in the instance map may be disconnected, so we want to
    # isolate each region of each instance
    label_map = measure.label(instance_map)

    # iterate over all regions
    for label in np.unique(label_map):
        # use this instead for a dense map (for visualization)
        # pixel_ids = np.where(label_map == label)
        # if not np.all(depth_map[pixel_ids] == -1):
        #     depth_map[pixel_ids] = np.mean(depth_map[pixel_ids][depth_map[pixel_ids] != -1])
        #     depth_map = np.nan_to_num(depth_map, nan=-1)

        # replace the depth of the region with the mean depth
        pixel_ids = np.where((label_map == label) & (is_valid(depth_map)))
        if not np.all(depth_map[pixel_ids] == -1):
            depth_map[pixel_ids] = np.mean(depth_map[pixel_ids])

    # return the new image and new depth map
    return instance_rgb, depth_map


def vertical_position_augment(image, instance_map, depth_map):
    # perform the vertical position augmentation
    # objects within the scene are copied to new locations with "significantly" different vertical position
    # depth values are copied to the new location

    # get the candidate objects for the augmentation
    object_ids, mask = identify_objects_to_augment(instance_map)

    # use only the NUM_OBJECTS_TO_AUGMENT largest objects
    for i, object_id in enumerate(object_ids):
        if i >= NUM_OBJECTS_TO_AUGMENT:
            break

        # get the pixels of this object
        pixel_ids = np.where(mask == object_ids[i])
        # get the centroid of this object
        centroid = np.array([np.mean(dimension) for dimension in pixel_ids])

        # change vertical position

        crop_percent = 0.15  # new centroid should not be within this percentage of borders
        min_vertical_change = 0.25  # make sure vertical position changes by at least 30% of the height
        assert 0.5 - crop_percent > min_vertical_change + 0.05  # cannot make the minimum impossible

        difference = [0, 0]
        # calculate a new centroid; make sure vertical position changed enough
        while np.abs(difference[0]) / mask.shape[0] < min_vertical_change:
            new_centroid = np.array([np.random.randint(dimension_size * crop_percent, dimension_size - dimension_size * crop_percent)
                                     for dimension_size in mask.shape])
            difference = (new_centroid - centroid).astype(int)

        # calculate where the object will move to by adding the difference between the current centroid and the new
        # centroid
        new_pixel_ids = np.vstack(pixel_ids).T + difference

        # remove pixels out of bounds
        indices = np.where((new_pixel_ids[:, 0] > 0) & (new_pixel_ids[:, 0] < image.shape[0]) & (new_pixel_ids[:, 1] > 0) & (new_pixel_ids[:, 1] < image.shape[1]))
        pixel_ids = np.vstack(pixel_ids).T[indices]
        new_pixel_ids = new_pixel_ids[indices]
        pixel_ids = tuple(pixel_ids.T)
        new_pixel_ids = tuple(new_pixel_ids.T)

        # change image and depth map
        image[new_pixel_ids] = image[pixel_ids]
        depth_map[new_pixel_ids] = depth_map[pixel_ids]

    # return the new image, the new depth map, and the candidate mask (for visualization)
    return image, depth_map, mask


def apparent_size_augment(image, instance_map, depth_map):
    """
    Perform the apparent size augmentation.
    Objects within the image are scaled, and their associated depths are scaled inversely.

    Z = (f / h) * H
    Z = depth
    f = focal length
    h = apparent size
    H = true size

    Change in apparent size is proportional to change in depth.
    No need for true size or focal length.
    """

    # get the candidate objects for the augmentation
    object_ids, mask = identify_objects_to_augment(instance_map)

    # use only the NUM_OBJECTS_TO_AUGMENT largest objects
    for i, object_id in enumerate(object_ids):
        if i >= NUM_OBJECTS_TO_AUGMENT:
            break

        # get the pixels of this object
        pixel_ids = np.where(mask == object_ids[i])
        # # get the centroid of this object
        # centroid = np.array([np.mean(dimension) for dimension in pixel_ids])
        # get the bbox of this object (row1, col1, row2, col2)
        row_min = pixel_ids[0].min()
        col_min = pixel_ids[1].min()
        bbox = (row_min, col_min, pixel_ids[0].max(), pixel_ids[1].max())
        # get the center of this object's bbox (row, col)
        center = ((bbox[2] + bbox[0]) // 2, (bbox[3] + bbox[1]) // 2)

        rel_pixel_ids = (pixel_ids[0] - row_min, pixel_ids[1] - col_min)
        shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1, 4)

        # get a rectangular image of just the object
        # pixels where the object does not exist have a color of (0, 0, 0, 0), which is transparent
        im_object = np.zeros(shape).astype(int)
        im_object_temp = image[pixel_ids]
        # add alpha channel
        im_object_temp = np.hstack((im_object_temp,
                                    np.expand_dims(np.full(im_object_temp.shape[0], 255), axis=1)))
        im_object[rel_pixel_ids] = im_object_temp

        # get the depth map for the object; -1s outside of object bounds
        depth_map_object = np.full(shape[:-1], -1).astype(float)
        depth_map_object[rel_pixel_ids] = depth_map[pixel_ids]

        # scale the object
        scale_percent = np.random.randint(125, 200)  # percent of original size
        width = int(im_object.shape[1] * scale_percent / 100)
        height = int(im_object.shape[0] * scale_percent / 100)
        dim = (width, height)
        im_object = cv2.resize(im_object.astype(np.uint8), dim, interpolation=cv2.INTER_NEAREST)

        # scale the depth map
        depth_map_object = cv2.resize(depth_map_object, dim, interpolation=cv2.INTER_NEAREST)

        # get the top left (x, y) of where the changed object will be
        top_left = (center[1] - dim[0] // 2, center[0] - dim[1] // 2)

        # paste the object onto the original image
        image = Image.fromarray(image)
        im_object = Image.fromarray(im_object)
        image.paste(im_object, top_left, mask=im_object)
        image = np.array(image)

        # calculate the new depth by dividing by the scale factor
        depth_map_object[is_valid(depth_map_object)] /= (scale_percent / 100)

        # paste the depth map onto the original depth map
        # areas with the object will be overwritten; gaps and areas without the object will retain original depth
        depth_map = Image.fromarray(depth_map)
        depth_map_object = Image.fromarray(depth_map_object)
        depth_map.paste(depth_map_object, top_left, mask=Image.fromarray(np.array(im_object)[:, :, 3] == 255))
        depth_map = np.asarray(depth_map)

    # return the new image and the new depth map
    return image, depth_map
