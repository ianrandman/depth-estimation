# Increasing Robustness of Monocular Depth Estimation with Pictorial Cues

## Prerequisites

Install requirements using `pip install -r requirements.txt`.

In the main directory, there must be `data_semantics` from the [KITTI Semantic Dataset](http://www.cvlibs.net/datasets/kitti/eval_instance_seg.php?benchmark=instanceSeg2015).
In this directory must exist `train_mapping.txt` from the development kit.

Also, `data_depth_annotated` must be in the same directory from the [Kitti Depth Dataset](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction).

## Augmentations

Run `python main.py` to produce the folder `augmented` with all the data to be used for the model.
Included is the original image and depth map, the augmented images and their associated depth maps, and the semantic 
rgb image.

## Examples:

To use the model, see the modified Monodepth2 project and associated `README.md`.
Train/val/test files must be created in `main.py`. They will be output in `augmented`.
These must be copied to `splits/custom` in the Monodepth2 project.

### Training

```
python train.py --data_path "/path/to/augmented" --dataset kitti_custom --split custom --frame_ids 0 --save_frequency 50 --num_epochs 100 --model_name vertical --aug vertical
```

The `--aug` option can go unused to only train on original data.

### Testing a single image (or multiple images)

```
python test_simple.py --model_path "path/to/vertical" --image_path "path/to/test_images/"
```

The image path can be either to a folder or a specific image.

### Evaluating a model

```
python evaluate_depth.py --data_path "path/to/augmented" --model_path "path/to/vertical" --eval_mono --dataset kitti_custom --eval_split custom
```