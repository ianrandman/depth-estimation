import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

from augmenter import instance_segmentation_augment, vertical_position_augment
from utils.file_utils import file_list
from utils.labels import id2label
from utils.read_depth import depth_read
from utils.interpolate_background import interpolate_background

SEMANTIC_DIR = os.path.join(os.path.dirname(__file__), 'data_semantics')
SEMANTIC_MAPPING = os.path.join(SEMANTIC_DIR, 'train_mapping.txt')
SEMANTIC_TRAIN_DIR = os.path.join(SEMANTIC_DIR, 'training')
SEMANTIC_TEST_DIR = os.path.join(SEMANTIC_DIR, 'testing')
SEMANTIC_TRAIN_IMAGE_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'image_2')
SEMANTIC_TRAIN_INSTANCE_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'instance')
SEMANTIC_TRAIN_SEMANTIC_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'semantic')
SEMANTIC_TEST_IMAGE_DIR = os.path.join(SEMANTIC_TEST_DIR, 'image_2')
SEMANTIC_TEST_INSTANCE_DIR = os.path.join(SEMANTIC_TEST_DIR, 'instance')

DEPTH_IMAGE_DIR = os.path.join(os.path.dirname(__file__),
                               'data_depth_selection/depth_selection/val_selection_cropped/image')
DEPTH_TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'data_depth_annotated/train')
DEPTH_TRAIN_RUNS = os.listdir(DEPTH_TRAIN_DIR)
DEPTH_VAL_DIR = os.path.join(os.path.dirname(__file__), 'data_depth_annotated/val')
DEPTH_VAL_RUNS = os.listdir(DEPTH_VAL_DIR)


def depth_test():
    image_dir = os.path.join(DEPTH_TRAIN_DIR, os.listdir(DEPTH_TRAIN_DIR)[1], 'proj_depth/groundtruth/image_02')
    for filename in os.listdir(image_dir):
        depth_image_path = os.path.join(image_dir, filename)
        depth_image = depth_read(depth_image_path)
        # interpolate_background(depth_image)

        plt.imshow(depth_image, cmap='magma')
        plt.show()
        x=1
        break

import matplotlib as mpl
mpl.use('TkAgg')

def main():
    semantic_train_mapping = file_list(SEMANTIC_MAPPING)
    # semantic_train_mapping_list = [mapping.split() for mapping in file_list(SEMANTIC_MAPPING)]
    # semantic_train_mapping = {mapping[2]: mapping[1] for mapping in semantic_train_mapping_list}
    #
    semantic_train_filenames = os.listdir(SEMANTIC_TRAIN_IMAGE_DIR)
    semantic_train_filename_mapping = list(zip(semantic_train_filenames, semantic_train_mapping))
    num_train = 0
    num_val = 0
    for filename, mapping in semantic_train_filename_mapping:
        if len(mapping) == 0:
            continue

        _, run, frame_num = mapping.split()
        if run in DEPTH_TRAIN_RUNS:
            depth_gt_filepath = os.path.join(DEPTH_TRAIN_DIR, run,
                                             'proj_depth/groundtruth/image_02', frame_num + '.png')
            num_train += 1
        elif run in DEPTH_VAL_RUNS:
            depth_gt_filepath = os.path.join(DEPTH_VAL_DIR, run,
                                             'proj_depth/groundtruth/image_02', frame_num + '.png')
            num_val += 1
        else:
            raise

        # load in raw data
        image = cv2.imread(os.path.join(SEMANTIC_TRAIN_IMAGE_DIR, filename))
        depth_map = depth_read(depth_gt_filepath)
        instance_map = cv2.imread(os.path.join(SEMANTIC_TRAIN_INSTANCE_DIR, filename), -1)
        semantic_map = cv2.imread(os.path.join(SEMANTIC_TRAIN_SEMANTIC_DIR, filename), -1)

        # instance segmentation augmentation
        instance_aug_image, instance_aug_depth_map = instance_segmentation_augment(instance_map, depth_map)

        # vertical position augmentation
        # TODO use original depth map
        vertical_aug_image, vertical_aug_depth_map = \
            vertical_position_augment(image.copy(), instance_map, instance_aug_depth_map.copy())

        plt.figure(dpi=500)
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.subplot(2, 2, 2)
        plt.imshow(vertical_aug_image)
        plt.subplot(2, 2, 3)
        plt.imshow(vertical_aug_depth_map)
        # plt.subplot(2, 2, 4)
        # plt.imshow(mask)
        plt.show()

    x = 1
    # semantic_test_numbers = ['0000' + filename.split('.')[0].split('_')[0] for filename in os.listdir(
    #     SEMANTIC_TEST_IMAGE_DIR)]
    #
    # for number in semantic_train_numbers:
    #     assert number in semantic_train_mapping
    # for number in semantic_test_numbers:
    #     assert number in semantic_train_mapping

# import matplotlib as mpl
# mpl.use('TkAgg')
# plt.ticklabel_format(useOffset=False)
def main2():
    for filename in os.listdir(SEMANTIC_TRAIN_IMAGE_DIR):
        image = cv2.imread(os.path.join(SEMANTIC_TRAIN_IMAGE_DIR, filename))

        instance_both = cv2.imread(os.path.join(SEMANTIC_TRAIN_INSTANCE_DIR, filename), -1)
        instance_rgb = instance_segmentation_augment(instance_both)
        # instance_semantic, instance_id = np.apply_along_axis(convert, 0, instance_both)
        # id_to_color_func = np.vectorize(lambda id: id2label[id].color)
        # instance_rgb = np.dstack(id_to_color_func(instance_semantic))

        semantic = cv2.imread(os.path.join(SEMANTIC_TRAIN_SEMANTIC_DIR, filename), -1)
        id_to_color_func = np.vectorize(lambda id: id2label[id].color)
        semanic_rgb = np.dstack(id_to_color_func(semantic))

        plt.figure(dpi=500)
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.subplot(2, 2, 2)
        plt.imshow(instance_rgb)
        plt.subplot(2, 2, 3)
        plt.imshow(semanic_rgb)
        plt.subplot(2, 2, 4)
        plt.imshow(instance_both)
        plt.show(block=True)


if __name__ == '__main__':
    main()
    # depth_test()
