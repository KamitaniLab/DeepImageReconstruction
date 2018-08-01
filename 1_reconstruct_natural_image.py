# Here the codes are used to reconstruct natural image from the CNN features decoded from brain;
# conditions:
# ROI: VC
# layers: use all conv and fc layers
# with DGN


# import
import os
import pickle
from datetime import datetime

import numpy as np
import PIL.Image
import scipy.io as sio

import caffe
from icnn.icnn_dgn_gd import reconstruct_image  # with DGN
from icnn.utils import clip_extreme_value, estimate_cnn_feat_std, normalise_img

# average image of ImageNet
img_mean_fn = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_fn)
img_mean = np.float32(
    [img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# load cnn model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2, 1, 0)
net = caffe.Classifier(prototxt_file, model_file,
                       mean=img_mean, channel_swap=channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1, 3, h, w)

# load the generator net
model_file = './net/generator_for_inverting_fc7/generator.caffemodel'
prototxt_file = './net/generator_for_inverting_fc7/generator.prototxt'
net_gen = caffe.Net(prototxt_file, model_file, caffe.TEST)
input_layer_gen = 'feat'  # input layer for generator net
output_layer_gen = 'deconv0'  # output layer for generator net

# feature size for input layer of the generator net
feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
num_of_unit = net_gen.blobs[input_layer_gen].data.size

# upper bound for input layer of the generator net
bound_file = './data/act_range/3x/fc7.txt'
upper_bound = np.loadtxt(bound_file, delimiter=' ',
                         usecols=np.arange(0, num_of_unit), unpack=True)
upper_bound = upper_bound.reshape(feat_size_gen)

# initial features for the input layer of the generator (we use a 0 vector as initial features)
initial_gen_feat = np.zeros_like(net_gen.blobs[input_layer_gen].data[0])

# feature std estimated from true CNN features of 10000 images
feat_std_file = './data/estimated_vgg19_cnn_feat_std.mat'
feat_std0 = sio.loadmat(feat_std_file)

# decoded CNN features
# the directory where all the decoded CNN features are saved
feat_dir = '/home/mu/aoki/work/deeprecon-datapub/decodedfeatures'

stim_type = 'natural'  # for the natural image

net_name = 'VGG19'

layers = []
for layer in net.blobs.keys():
    if 'conv' in layer or 'fc' in layer:  # use all conv and fc layers
        layers.append(layer)

subjects = ['S1', 'S2', 'S3']  # subjects: ['S1','S2','S3']

ROIs = ['VC']

# the image used in Figure 2(a), the image label can be from 1 to 50 for natural images in test data
image_label_list = [4, 9, 2, 36, 45, 16]

# make folder for saving the results
save_dir = './result'
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir, save_folder)
os.mkdir(save_path)

# reconstruction options
opts = {

    'loss_type': 'l2',  # the loss function type: {'l2','l1','inner','gram'}

    'iter_n': 200,  # the total number of iterations for gradient descend

    'disp_every': 1,  # display the information on the terminal for every n iterations

    'lr_start': 2.,  # learning rate
    'lr_end': 1e-10,

    'momentum_start': 0.9,  # gradient with momentum
    'momentum_end': 0.9,

    # decay for the features of the input layer of the generator after each iteration
    'decay_start': 0.01,
    'decay_end': 0.01,

    # name of the input layer of the generator (str)
    'input_layer_gen': input_layer_gen,
    # name of the output layer of the generator (str)
    'output_layer_gen': output_layer_gen,

    # set the upper boundary for the input layer of the generator
    'feat_upper_bound': upper_bound,
    'feat_lower_bound': 0.,  # set the lower boundary for the input layer of the generator

    # the initial features of the input layer of the generator (setting to None will use random noise as initial features)
    'initial_gen_feat': initial_gen_feat,

}

# save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path, save_name), 'w') as f:
    pickle.dump(opts, f)
    f.close()

# reconstruction
for subject in subjects:  # loop for subjects
    for ROI in ROIs:  # loop for ROIs
        for image_label in image_label_list:  # loop for image label
            #
            print('')
            print('subject: '+subject)
            print('ROI: '+ROI)
            print('image_label: '+str(image_label))
            print('')

            # load the decoded CNN features
            features = {}
            for layer in layers:
                file_name = os.path.join(feat_dir, stim_type, net_name, layer, subject, ROI, stim_type+'-'+net_name+'-'+layer+'-'+subject +
                                         '-'+ROI+'-Img%04d.mat' % image_label)  # the file full name depends on the data structure for decoded CNN features
                feat = sio.loadmat(file_name)['feat']
                if 'fc' in layer:
                    num_of_unit = feat.size
                    feat = feat.reshape(num_of_unit)
                features[layer] = feat

            # correct the norm of the decoded CNN features
            for layer in layers:
                feat = features[layer]
                feat_std = estimate_cnn_feat_std(feat)
                feat = (feat / feat_std) * feat_std0[layer]
                features[layer] = feat

            # weight of each layer in the total loss function
            num_of_layer = len(layers)
            feat_norm_list = np.zeros(num_of_layer, dtype='float32')
            for j, layer in enumerate(layers):
                # norm of the CNN features for each layer
                feat_norm_list[j] = np.linalg.norm(features[layer])
            # use the inverse of the squared norm of the CNN features as the weight for each layer
            weights = 1. / (feat_norm_list**2)
            # normalise the weights such that the sum of the weights = 1
            weights = weights / weights.sum()
            layer_weight = {}
            for j, layer in enumerate(layers):
                layer_weight[layer] = weights[j]
            opts['layer_weight'] = layer_weight

            # reconstruction
            recon_img, loss_list = reconstruct_image(
                features, net, net_gen, **opts)

            # save the results
            save_name = 'recon_img' + '_' + subject + \
                '_' + ROI + '_Img%04d.mat' % image_label
            # save the raw reconstructed image
            sio.savemat(os.path.join(save_path, save_name),
                        {'recon_img': recon_img})

            # to better display the image, clip pixels with extreme values (0.02% of pixels with extreme low values and 0.02% of the pixels with extreme high values).
            # and then normalise the image by mapping the pixel value to be within [0,255].
            save_name = 'recon_img' + '_' + subject + \
                '_' + ROI + '_Img%04d.jpg' % image_label
            PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img, pct=0.04))).save(
                os.path.join(save_path, save_name))

# end
