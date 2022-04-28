"""
The main file for performing data augmentations on the KITTI dataset.

__author__ = Ian Randman
"""

import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from augmenter import instance_segmentation_augment, vertical_position_augment, apparent_size_augment
from utils.file_utils import file_list, depth_read

SEMANTIC_DIR = os.path.join(os.path.dirname(__file__), 'data_semantics')
SEMANTIC_MAPPING = os.path.join(SEMANTIC_DIR, 'train_mapping.txt')
SEMANTIC_TRAIN_DIR = os.path.join(SEMANTIC_DIR, 'training')
SEMANTIC_TEST_DIR = os.path.join(SEMANTIC_DIR, 'testing')
SEMANTIC_TRAIN_IMAGE_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'image_2')
SEMANTIC_TRAIN_INSTANCE_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'instance')
SEMANTIC_TRAIN_SEMANTIC_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'semantic')
SEMANTIC_TRAIN_SEMANTIC_RGB_DIR = os.path.join(SEMANTIC_TRAIN_DIR, 'semantic_rgb')
SEMANTIC_TEST_IMAGE_DIR = os.path.join(SEMANTIC_TEST_DIR, 'image_2')
SEMANTIC_TEST_INSTANCE_DIR = os.path.join(SEMANTIC_TEST_DIR, 'instance')

DEPTH_IMAGE_DIR = os.path.join(os.path.dirname(__file__),
                               'data_depth_selection/depth_selection/val_selection_cropped/image')
DEPTH_TRAIN_DIR = os.path.join(os.path.dirname(__file__), 'data_depth_annotated/train')
DEPTH_TRAIN_RUNS = os.listdir(DEPTH_TRAIN_DIR)
DEPTH_VAL_DIR = os.path.join(os.path.dirname(__file__), 'data_depth_annotated/val')
DEPTH_VAL_RUNS = os.listdir(DEPTH_VAL_DIR)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'augmented')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


def main():
    # get a mapping from the semantic dataset to the original dataset
    semantic_train_mapping = file_list(SEMANTIC_MAPPING)
    semantic_train_filenames = os.listdir(SEMANTIC_TRAIN_IMAGE_DIR)
    semantic_train_filename_mapping = list(zip(semantic_train_filenames, semantic_train_mapping))
    num_train = 0
    num_val = 0

    # iterate over the mapping
    for filename, mapping in tqdm(semantic_train_filename_mapping):
        if len(mapping) == 0:
            continue

        _, run, frame_num = mapping.split()

        # not every image in the semantic dataset has a mapping to the full dataset
        # get the ground truth from the depth dataset
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
        semantic_rgb_image = cv2.imread(os.path.join(SEMANTIC_TRAIN_SEMANTIC_RGB_DIR, filename), -1)

        # instance segmentation augmentation
        instance_aug_image, instance_aug_depth_map = instance_segmentation_augment(instance_map, depth_map.copy())

        # vertical position augmentation
        vertical_position_aug_image, vertical_position_aug_depth_map, mask = \
            vertical_position_augment(image.copy(), instance_map, depth_map.copy())

        # apparent size augmentation
        apparent_size_aug_image, apparent_size_aug_depth_map = \
            apparent_size_augment(image.copy(), instance_map, depth_map.copy())

        # uncomment to show the augmentations
        # plt.figure(dpi=700)
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.imshow(instance_aug_image)
        # plt.subplot(2, 2, 3)
        # plt.imshow(instance_aug_depth_map, cmap='magma')
        # plt.show()
        #
        # plt.figure(dpi=700)
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.imshow(vertical_position_aug_image)
        # plt.subplot(2, 2, 3)
        # plt.imshow(vertical_position_aug_depth_map, cmap='magma')
        # plt.subplot(2, 2, 4)
        # plt.imshow(mask)
        # plt.show()
        #
        # plt.figure(dpi=700)
        # plt.subplot(2, 2, 1)
        # plt.imshow(image)
        # plt.subplot(2, 2, 2)
        # plt.imshow(apparent_size_aug_image)
        # plt.subplot(2, 2, 3)
        # plt.imshow(apparent_size_aug_depth_map, cmap='magma')
        # plt.subplot(2, 2, 4)
        # plt.imshow(mask)
        # plt.show()

        # make directory for this frame
        dir_path = os.path.join(OUT_DIR, f'{run}_frame_{frame_num}')
        os.makedirs(dir_path, exist_ok=True)

        # filepaths of output images
        image_filepath = os.path.join(dir_path, 'image.jpg')
        semantic_rgb_filepath = os.path.join(dir_path, 'semantic_rgb.jpg')
        instance_aug_image_filepath = os.path.join(dir_path, 'instance_aug_image.jpg')
        vertical_position_aug_image_filepath = os.path.join(dir_path, 'vertical_position_aug_image.jpg')
        apparent_size_aug_image_filepath = os.path.join(dir_path, 'apparent_size_aug_image.jpg')

        # filepaths of output depth maps
        depth_map_filepath = os.path.join(dir_path, 'depth_map.png')
        instance_aug_depth_map_filepath = os.path.join(dir_path, 'instance_aug_depth_map.png')
        vertical_position_aug_depth_map_filepath = os.path.join(dir_path, 'vertical_position_aug_depth_map.png')
        apparent_size_aug_depth_map_filepath = os.path.join(dir_path, 'apparent_size_aug_depth_map.png')

        # save the images
        cv2.imwrite(image_filepath, image)
        cv2.imwrite(semantic_rgb_filepath, semantic_rgb_image)
        cv2.imwrite(instance_aug_image_filepath, instance_aug_image)
        cv2.imwrite(vertical_position_aug_image_filepath, vertical_position_aug_image)
        cv2.imwrite(apparent_size_aug_image_filepath, apparent_size_aug_image)

        # save the depth maps
        cv2.imwrite(depth_map_filepath, depth_map)
        cv2.imwrite(instance_aug_depth_map_filepath, instance_aug_depth_map)
        cv2.imwrite(vertical_position_aug_depth_map_filepath, vertical_position_aug_depth_map)
        cv2.imwrite(apparent_size_aug_depth_map_filepath, apparent_size_aug_depth_map)

    # uncomment to produce train/val/test split files
    # from sklearn.model_selection import train_test_split
    # train_dirs, test_dirs = train_test_split(os.listdir(OUT_DIR), train_size=0.8)
    # # train_dirs, val_dirs = train_test_split(os.listdir(OUT_DIR), train_size=0.9)
    # with open(os.path.join(OUT_DIR, 'train_files.txt'), 'w') as f:
    #     for dir in train_dirs:
    #         f.write(dir + '\n')
    # with open(os.path.join(OUT_DIR, 'val_files.txt'), 'w') as f:
    #     f.write('\n')
    # with open(os.path.join(OUT_DIR, 'test_files.txt'), 'w') as f:
    #     for dir in test_dirs:
    #         f.write(dir + '\n')


if __name__ == '__main__':
    main()
