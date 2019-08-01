'''
Reconstruct colored artificial shapes from CNN features decoded from multiple visual areas in the brain.

- ROI: V1, V2, V3, V4, HVC, VC
- Layers: all conv and fc layers
- Reconstruction algorithm: Without DGN + LBFGS
'''


import os
import pickle
from datetime import datetime
from itertools import product

import caffe
import numpy as np
import PIL.Image
import scipy.io as sio

from icnn.icnn_lbfgs import reconstruct_image  # Without DGN
from icnn.utils import clip_extreme_value, estimate_cnn_feat_std, normalise_img


# Settings ###################################################################

# GPU usage settings
caffe.set_mode_gpu()
caffe.set_device(0)

# Decoded features settings
decoded_features_dir = './data/decodedfeatures'
decode_feature_filename = lambda net, layer, subject, roi, image_type, image_label: os.path.join(decoded_features_dir, image_type, net, layer, subject, roi,
                                                                                                 '%s-%s-%s-%s-%s-%s.mat' % (image_type, net, layer, subject, roi, image_label))

# Data settings
results_dir = './results'

subjects_list = ['S1', 'S2', 'S3']

rois_list = ['V1', 'V2', 'V3', 'V4', 'HVC', 'VC']

network = 'VGG19'

# Images in figure 3A
image_type = 'color_shape'
image_label_list = ['Img0004',
                    'Img0008',
                    'Img0011']

max_iteration = 200


# Main #######################################################################

# Initialize CNN -------------------------------------------------------------

# Average image of ImageNet
img_mean_file = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_file)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# CNN model
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

# CNN Layers (all conv and fc layers)
layers = [layer for layer in net.blobs.keys() if 'conv' in layer or 'fc' in layer]

# Setup results directory ----------------------------------------------------

save_dir_root = os.path.join(results_dir, os.path.splitext(__file__)[0])
if not os.path.exists(save_dir_root):
    os.makedirs(save_dir_root)

# Set reconstruction options -------------------------------------------------

opts = {
    # The loss function type: {'l2','l1','inner','gram'}
    'loss_type': 'l2',

    # The maximum number of iterations
    'maxiter': max_iteration,

    # The initial image for the optimization (setting to None will use random noise as initial image)
    'initial_image': initial_image,

    # Display the information on the terminal or not
    'disp': True
}

# Save the optional parameters
with open(os.path.join(save_dir_root, 'options.pkl'), 'w') as f:
    pickle.dump(opts, f)

# Reconstrucion --------------------------------------------------------------

for subject, roi, image_label in product(subjects_list, rois_list, image_label_list):

    print('')
    print('Subject:     ' + subject)
    print('ROI:         ' + roi)
    print('Image label: ' + image_label)
    print('')

    save_dir = os.path.join(save_dir_root, subject, roi)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load the decoded CNN features
    features = {}
    for layer in layers:
        # The file full name depends on the data structure for decoded CNN features
        file_name = decode_feature_filename(network, layer, subject, roi, image_type, image_label)
        feat = sio.loadmat(file_name)['feat']
        if 'fc' in layer:
            feat = feat.reshape(feat.size)

        # Correct the norm of the decoded CNN features
        feat_std = estimate_cnn_feat_std(feat)
        feat = (feat / feat_std) * feat_std0[layer]

        features.update({layer: feat})

    # Weight of each layer in the total loss function

    # Norm of the CNN features for each layer
    feat_norm = np.array([np.linalg.norm(features[layer]) for layer in layers], dtype='float32')

    # Use the inverse of the squared norm of the CNN features as the weight for each layer
    weights = 1. / (feat_norm ** 2)

    # Normalise the weights such that the sum of the weights = 1
    weights = weights / weights.sum()
    layer_weight = dict(zip(layers, weights))

    opts.update({'layer_weight': layer_weight})

    # Reconstruction
    snapshots_dir = os.path.join(save_dir, 'snapshots', 'image-%s' % image_label)
    recon_img, loss_list = reconstruct_image(features, net,
                                             save_intermediate=True,
                                             save_intermediate_path=snapshots_dir,
                                             **opts)

    # Save the results

    # Save the raw reconstructed image
    save_name = 'recon_img' + '-' + image_label + '.mat'
    sio.savemat(os.path.join(save_dir, save_name), {'recon_img': recon_img})

    # To better display the image, clip pixels with extreme values (0.02% of
    # pixels with extreme low values and 0.02% of the pixels with extreme high
    # values). And then normalise the image by mapping the pixel value to be
    # within [0,255].
    save_name = 'recon_img_normalized' + '-' + image_label + '.jpg'
    PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(os.path.join(save_dir, save_name))

print('Done')
