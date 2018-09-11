# Deep Image Reconstruction

Data and demo codes for [Shen, Horikawa, Majima, & Kamitani (2017). Deep image reconstruction from human brain activity. bioRxiv](https://www.biorxiv.org/content/early/2017/12/30/240317).

## Requirements

- Python 2.7
- [icnn](https://github.com/KamitaniLab/icnn)
- Numpy
- Scipy
- Pillow (PIL)
- Caffe with up-convolutional layer
    - https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim)
    - Both CPU and GPU installation are OK

## Usage

### Preparation

1. Download data files from figshare (see [data/README.md](data/README.md)).
2. Download Caffe networks (see [net/README.md](net/README.md)).

### CNN feature decoding from brain activity

You can skip the feature decoding from brain activity since we provide the decoded CNN features used in the original paper (see [data/README.md](data/README.md)).

We used the same methodology in our previous study for the feature decoding ([Horikawa & Kamitani, 2017, Generic decoding of seen and imagined objects using hierarchical visual features, Nat Commun.](https://www.nature.com/articles/ncomms15037)).
Demo programs for Matlab and Python are available at <https://github.com/KamitaniLab/GenericObjectDecoding>.

### Image reconstruction from decoded CNN features

We provide seven scripts that reproduce main figures in the original paper.

- 1_reconstruct_natural_image.py
    - Reconstructing natural images from the CNN features decoded from the brain with deep generator network (DGN); reproducing results in Figure 2(a)
- 2_reconstruct_natural_image_without_DGN.py
    - Reconstructing natural images from CNN features decoded from the brain without deep generator network (DGN); reproducing results in Figure 2(b)
- 3_reconstruct_natural_image_different_combinations_of_CNN_layers.py
    - Reconstructing natural images from CNN features decoded from the brain with different combinations of CNN layers; reproducing results in Figure 2(d)
- 4_reconstruct_shape_image.py
    - Reconstructing colored artificial shapes from CNN features decoded from the brain; reproducing results in Figure 3(a)
- 5_reconstruct_shape_image_different_ROI.py
    - Reconstructing colored artificial shapes from CNN features decoded from multiple visual areas in the brain; reproducing results in Figure 3(c)
- 6_reconstruct_alphabet_image.py
    - Reconstructing alphabetical letters shapes from CNN features decoded from the brain; reproducing results in Figure 3(e)
- 7_reconstruct_imagined_image.py
    - Reconstructing imagined image from CNN features decoded from the brain; reproducing results in Figure 4

## Data

- fMRI data: [Deep Image Reconstruction@OpenNeuro](https://openneuro.org/datasets/ds001506)
- Decoded CNN features: [Deep Image Reconstruction@figshare](https://figshare.com/articles/Deep_Image_Reconstruction/7033577)

## Notes

### Enable back-propagation in the CNNs

In the demo code, we use pre-trained VGG19 model (http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel) and pre-trained deep generator network (DGN; https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip; [Dosovitskiy & Brox, 2016, Generating Images with Perceptual Similarity Metrics based on Deep Networks. arXiv.](https://arxiv.org/abs/1602.02644)).
In order to make back-propagation work, one line should be added to the prototxt files (the file describes the configuration of the CNN model):

`force_backward: true`.

### CNN features before or after ReLU

In our study, we define CNN features of conv layers or fc layers as the output immediately after the convolutional or fully-connected computation, before applying the Rectified-Linear-Unit (ReLU).
However, as default setting, ReLU operation is an in-place computation, which will override the CNN features we need.
In order to use the CNN features before the ReLU operation, we need to modify the prototxt file.
Taking the VGG19 prototxt file as an example:

In the original prototxt file, ReLU is in-place computation:

```
layers {
  bottom: "conv1_1"
  top: "conv1_1"
  name: "relu1_1"
  type: RELU
}
```

Now, we modify it as:

```
layers {
  bottom: "conv1_1"
  top: "relu1_1"
  name: "relu1_1"
  type: RELU
}
```
