##Information

---
name: 19-layer model from the arXiv paper - "Very Deep Convolutional Networks for Large-Scale Image Recognition"

caffemodel: VGG_ILSVRC_19_layers

caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

license: http://creativecommons.org/licenses/by-nc/4.0/ (non-commercial use only)

caffe_version: trained using a custom Caffe-based framework

gist_id: 3785162f95cd2d5fee77
---

## Description

The model is an improved version of the 19-layer model used by the VGG team in the ILSVRC-2014 competition. The details can be found in the following [arXiv paper](http://arxiv.org/pdf/1409.1556),

    Very Deep Convolutional Networks for Large-Scale Image Recognition
    K. Simonyan, A. Zisserman
    arXiv:1409.1556

Please cite the paper if you use the model.

In the paper, the model is denoted as the configuration `E` trained with scale jittering. The input images should be zero-centered by mean pixel (rather than mean image) subtraction. Namely, the following BGR values should be subtracted: `[103.939, 116.779, 123.68]`.

## Caffe compatibility

The models are currently supported by the `dev` branch of [Caffe](https://github.com/BVLC/caffe/), but are not yet compatible with `master`.
An example of how to use the models in Matlab can be found in [matlab/caffe/matcaffe_demo_vgg.m](https://github.com/BVLC/caffe/blob/dev/matlab/caffe/matcaffe_demo_vgg_mean_pix.m) 

## ILSVRC-2012 performance

Using dense single-scale evaluation (the smallest image side rescaled to 384), the top-5 classification error on the validation set of ILSVRC-2012 is 8.0% (see Table 3 in the [arXiv paper](http://arxiv.org/pdf/1409.1556)).

Using dense multi-scale evaluation (the smallest image side rescaled to 256, 384, and 512), the top-5 classification error is 7.5% on the validation set and 7.3% on the test set of ILSVRC-2012 (see Tables 4 and 6 in the [arXiv paper](http://arxiv.org/pdf/1409.1556)).