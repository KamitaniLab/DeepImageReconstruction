#!/bin/bash
#
# Download demo data for Deep Image Reconstruction
#

## Functions

function download_file () {
    dlurl=$1
    dlpath=$2
    dldir=$(dirname $dlpath)
    dlfile=$(basename $dlpath)

    [ -d $didir ] || mkdir $dldir
    if [ -f $dldir/$dlfile ]; then
        echo "$dlfile has already been downloaded."
    else
        curl -o $dldir/$dlfile $dlurl
    fi
}

## Main

download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12955064/decoded_vgg19_cnn_feat.mat decoded_vgg19_cnn_feat.mat
download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12955070/estimated_vgg19_cnn_feat_std.mat estimated_vgg19_cnn_feat_std.mat
download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12955067/estimated_vgg19LeakyReluAvePool_cnn_feat_std.mat estimated_vgg19LeakyReluAvePool_cnn_feat_std.mat
download_file https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12955073/ilsvrc_2012_mean.npy ilsvrc_2012_mean.npy
