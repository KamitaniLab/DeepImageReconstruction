# Here the codes are used to reconstruct natural image from the CNN features decoded from brain;
# Reconstructions using different combinations of CNN layers;
# conditions:
# ROI: VC
# layers: different combinations of CNN layers
# without DGN


# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from datetime import datetime

from icnn.utils import estimate_cnn_feat_std, clip_extreme_value, normalise_img
from icnn.icnn_lbfgs import reconstruct_image # without DGN

# average image of ImageNet
img_mean_fn = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_fn)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# load cnn model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2,1,0)
net = caffe.Classifier(prototxt_file, model_file, mean = img_mean, channel_swap = channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1,3,h,w)

# initial image for the optimization (here we use the mean of ilsvrc_2012_mean.npy as RGB values)
initial_image = np.zeros((h,w,3),dtype='float32')
initial_image[:,:,0] = img_mean[2].copy()
initial_image[:,:,1] = img_mean[1].copy()
initial_image[:,:,2] = img_mean[0].copy()

# feature std estimated from true CNN features of 10000 images
feat_std_file = './data/estimated_vgg19_cnn_feat_std.mat'
feat_std0 = sio.loadmat(feat_std_file)

# decoded CNN features
feat_dir = '/home/mu/aoki/work/deeprecon-datapub/decodedfeatures' # the directory where all the decoded CNN features are saved

stim_type = 'natural' # for the natural image

net_name = 'VGG19'

subjects = ['S1'] # subjects: ['S1','S2','S3']

ROIs = ['VC']

# layer set
layer_set1 = ['conv1_1','conv1_2']

layer_set2 = ['conv2_1','conv2_2']

layer_set3 = ['conv3_1','conv3_2','conv3_3','conv3_4']

layer_set4 = ['conv4_1','conv4_2','conv4_3','conv4_4']

layer_set5 = ['conv5_1','conv5_2','conv5_3','conv5_4']

layer_set6 = ['fc6']

layer_set7 = ['fc7']

layer_set8 = ['fc8']

# layer combinations
layer_combinations = {}

layer_combinations['layers_1to1'] = layer_set1

layer_combinations['layers_1to3'] = layer_set1 + layer_set2 + layer_set3

layer_combinations['layers_1to5'] = layer_set1 + layer_set2 + layer_set3 + layer_set4 + layer_set5

layer_combinations['layers_1to7'] = layer_set1 + layer_set2 + layer_set3 + layer_set4 + layer_set5 + layer_set6 + layer_set7

# image label
image_label_list = [16,36,42] # the image used in Figure 2(d), the image label can be from 1 to 50 for natural images in test data

# make folder for saving the results
save_dir = './result'
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)

# reconstruction options
opts = {
    
    'loss_type': 'l2', # the loss function type: {'l2','l1','inner','gram'}
    
    'maxiter': 200, # the maximum number of iterations
    
    'disp': True, # print or not the information on the terminal
    
    'initial_image': initial_image, # the initial image for the optimization (setting to None will use random noise as initial image)
    
    }

# save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path,save_name),'w') as f:
    pickle.dump(opts,f)
    f.close()

# reconstruction
for subject in subjects: # loop for subjects
    for ROI in ROIs: # loop for ROIs
        for image_label in image_label_list: # loop for image label
            for combination_name, layers in layer_combinations.iteritems(): # loop for layer_combinations
                #
                print('')
                print('subject: '+subject)
                print('ROI: '+ROI)
                print('image_label: '+str(image_label))
                print('layer_combination: '+combination_name)
                print('')
                
                # load decoded CNN features
                features = {}
                for layer in layers:
                    file_name = os.path.join(feat_dir,stim_type,net_name,layer,subject,ROI,stim_type+'-'+net_name+'-'+layer+'-'+subject+'-'+ROI+'-Img%04d.mat'%image_label) # the file full name depends on the data structure for decoded CNN features
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
                feat_norm_list = np.zeros(num_of_layer,dtype='float32')
                for j, layer in enumerate(layers):
                    feat_norm_list[j] = np.linalg.norm(features[layer]) # norm of the CNN features for each layer
                weights = 1. / (feat_norm_list**2) # use the inverse of the squared norm of the CNN features as the weight for each layer
                weights = weights / weights.sum() # normalise the weights such that the sum of the weights = 1
                layer_weight = {}
                for j, layer in enumerate(layers):
                    layer_weight[layer] = weights[j]
                opts['layer_weight'] = layer_weight
                
                # reconstruction
                recon_img, loss_list = reconstruct_image(features, net, **opts)
                
                # save the results
                save_name = 'recon_img' + '_' + subject + '_' + ROI + '_' + combination_name + '_Img%04d.mat'%image_label
                sio.savemat(os.path.join(save_path,save_name),{'recon_img':recon_img}) # save the raw reconstructed image
                
                # to better display the image, clip pixels with extreme values (0.02% of pixels with extreme low values and 0.02% of the pixels with extreme high values).
                # and then normalise the image by mapping the pixel value to be within [0,255].
                save_name = 'recon_img' + '_' + subject + '_' + ROI + '_' + combination_name + '_Img%04d.jpg'%image_label
                PIL.Image.fromarray(normalise_img(clip_extreme_value(recon_img,pct=0.04))).save(os.path.join(save_path,save_name))

# end
