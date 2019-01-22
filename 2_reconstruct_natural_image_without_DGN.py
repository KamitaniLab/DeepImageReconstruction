'''
Reconstruct natural images from CNN features decoded from the brain without deep generator network (DGN).

- ROI: VC
- Layers: all conv and fc layers
- Reconstruction algorithm: Without DGN
'''


import os
import pickle
from datetime import datetime
from itertools import product

import numpy as np
import PIL.Image
import scipy.io as sio
import caffe
from icnn.icnn_lbfgs import reconstruct_image  # without DGN
from icnn.utils import clip_extreme_value, estimate_cnn_feat_std, normalise_img


# Average image of ImageNet
img_mean_fn = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_fn)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# Load cnn model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2, 1, 0)
net = caffe.Classifier(prototxt_file, model_file, mean=img_mean, channel_swap=channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1, 3, h, w)

# Initial image for the optimization (here we use the mean of ilsvrc_2012_mean.npy as RGB values)
initial_image = np.zeros((h, w, 3), dtype='float32')
initial_image[:, :, 0] = img_mean[2].copy()
initial_image[:, :, 1] = img_mean[1].copy()
initial_image[:, :, 2] = img_mean[0].copy()

# Feature SD estimated from true CNN features of 10000 images
feat_std_file = './data/estimated_vgg19_cnn_feat_std.mat'
feat_std0 = sio.loadmat(feat_std_file)

# Decoded CNN features
feat_dir = './data/decodedfeatures' # the directory where all the decoded CNN features are saved
stim_type = 'natural'
net_name = 'VGG19'

# CNN Layers (all conv and fc layers)
layers = [layer for layer in net.blobs.keys()
          if 'conv' in layer or 'fc' in layer]

subjects = ['S1', 'S2', 'S3']

rois_list = ['VC']

# The image used in Figure 3A, the image label can be from 1 to 50 for natural images in test data
image_label_list = [2, 11, 45]

# Make folder for saving the results
save_dir = './result'
save_folder = __file__.split('.')[0] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir, save_folder)
os.mkdir(save_path)

# Reconstruction options
opts = {
    # The loss function type: {'l2','l1','inner','gram'}
    'loss_type': 'l2',

    # The maximum number of iterations
    'maxiter': 200,

    # The initial image for the optimization (setting to None will use random noise as initial image)
    'initial_image': initial_image,

    # Print the information on the terminal or not
    'disp': True
}

# Save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path, save_name), 'w') as f:
    pickle.dump(opts, f)

# Reconstruction
for subject, roi, image_label in product(subjects, rois_list, image_label_list):

    print('')
    print('Subject: ' + subject)
    print('ROI: ' + roi)
    print('Image label: '+ str(image_label))
    print('')

    # Load the decoded CNN features
    features = {}
    for layer in layers:
        # The file full name depends on the data structure for decoded CNN features
        file_name = os.path.join(feat_dir, stim_type, net_name, layer, subject, roi,
                                 stim_type + '-' + net_name + '-' + layer + '-' + subject + '-' + roi + '-Img%04d.mat' % image_label)
        feat = sio.loadmat(file_name)['feat']
        if 'fc' in layer:
            num_of_unit = feat.size
            feat = feat.reshape(num_of_unit)
        features[layer] = feat

    # Correct the norm of the decoded CNN features
    for layer in layers:
        feat = features[layer]
        feat_std = estimate_cnn_feat_std(feat)
        feat = (feat / feat_std) * feat_std0[layer]
        features[layer] = feat

    # Weight of each layer in the total loss function
    num_of_layer = len(layers)
    feat_norm_list = np.zeros(num_of_layer, dtype='float32')
    for j, layer in enumerate(layers):
        # Norm of the CNN features for each layer
        feat_norm_list[j] = np.linalg.norm(features[layer])
    # Use the inverse of the squared norm of the CNN features as the weight for each layer
    weights = 1. / (feat_norm_list**2)
    # Normalise the weights such that the sum of the weights = 1
    weights = weights / weights.sum()
    layer_weight = {}
    for j, layer in enumerate(layers):
        layer_weight[layer] = weights[j]
    opts['layer_weight'] = layer_weight

    # Reconstruction
    recon_img, loss_list = reconstruct_image(features, net, **opts)

    # Save the results
    save_name = 'recon_img' + '_' + subject + '_' + roi + '_Img%04d.mat' % image_label
    # Save the raw reconstructed image
    sio.savemat(os.path.join(save_path, save_name), {'recon_img': recon_img})

    # To better display the image, clip pixels with extreme values (0.02% of
    # pixels with extreme low values and 0.02% of the pixels with extreme high
    # values). And then normalise the image by mapping the pixel value to be
    # within [0,255].
    save_name = 'recon_img' + '_' + subject + '_' + roi + '_Img%04d.jpg' % image_label
    PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(os.path.join(save_path, save_name))
