# Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking (CVPR 2021)
Pytorch implementation of the ArTIST motion model. In this repo, there are

- Training [script](https://github.com/fatemeh-slh/ArTIST/blob/main/train_ae.py) for the Moving Agent network
- Training [script](https://github.com/fatemeh-slh/ArTIST/blob/main/train_ar.py) for the ArTIST motion model
- Demo [script](https://github.com/fatemeh-slh/ArTIST/blob/main/demo_scoring.py) for Inferring the likelihood of current observations (detections)
- Demo [script](https://github.com/fatemeh-slh/ArTIST/blob/main/demo_inpainting.py) for Inpainting the missing observation/detections


## Demo 1: Likelihood estimation of observation
Run:
```
python3 demo_scoring.py
```
This will generate the output in the `temp/ar/log_p` directory, look like this:
![scoring demo](https://github.com/fatemeh-slh/ArTIST/blob/main/temp/ar/log_p/likelihood.jpg)

This demo gets as input a pretrained model of the Moving Agent Network (MA-Net), a pretrained model of ArTIST, the centroids (obtain centroids via the [script](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/clustering.py) in the utils), a demo test sample index and the number of clusters.

The model then evaluates the log-likelihood (lower the better) of all detections as the continuation of the observed sequence. 

## Demo 2: Sequence inpainting
Run:
```
python3 demo_inpainting.py
```
This will generate the multiple plauusible continuations of an observed motion, stored in the `temp/ar/inpainting` directory. One example looks like this:
![inpainting demo](https://github.com/fatemeh-slh/ArTIST/blob/main/temp/ar/inpainting/inpainting_25.jpg)

This demo gets as input  a pretrained model of the Moving Agent Network (MA-Net), a pretrained model of ArTIST, the centroids (obtain centroids via the [script](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/clustering.py) in the utils), a demo test sample index and the number of samples we wish to generate.

For each generated future sequence, it computes the IoU between the last generated bounding box and the last groundtruth bounding box, as well as the mean IoU for the entire generated sequence and the groundtruth sequence.


### Utilities
In this repo, there are a number of scripts to generate the required data to train/evaluate ArTIST.
- [`prepare_data`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/prepare_data.py): Given the annotations of a dataset (e.g., MOT17), it extracts the motion sequences as well as the IDs of the social tracklets living the life span of the corresponding sequence, and stores it as a dictionary. If there are multiple tracking datasets that you wish to combine, you can use the `merge_datasets()` function inside this script.
- [`clustering`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/clustering.py): Given the output dictionary of `prepare_data` script, this script performs the K-Means clustering and stores the centroids which are then used in the ArTIST model.
- [`dataloader_ae`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/dataloader_ae.py) and [`dataloader_ar`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/dataloader_ar.py): Given the post-processes version of the dataset dictionary (which can be done by running the [`post_process`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/post_process.py) script), these two scripts define the dataloaders for training the MA-Net and ArTIST. Note that the dataloader of ArTIST uses the MA-Net to compute the social information. This can also be done jointly in an end-to-end fashion, which we observed almost no difference.
- [`create_demo_test_subset`](https://github.com/fatemeh-slh/ArTIST/blob/main/utils/create_demo_test_subset.py): In order to run the demo scripts, you need to run this script. However, the demo test subset has been produced and stored in [`data/demo_test_subset.npy`](https://github.com/fatemeh-slh/ArTIST/blob/main/data/demo_test_subset.npy).

### Data
You can download the required data from the [Release](https://github.com/fatemeh-slh/ArTIST/releases/tag/data-release) and put it in `data/` directory.

## Citation
If you find this work useful in your own research, please consider citing:
```
@inproceedings{saleh2021probabilistic,
author={Saleh, Fatemeh and Aliakbarian, Sadegh and Rezatofighi, Hamid and Salzmann, Mathieu and Gould, Stephen},
title = {Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking},
booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
year = {2021}
}
```
