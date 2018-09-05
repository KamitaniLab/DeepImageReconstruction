#!/bin/bash
#
# Download DNNs
#
# Usage:
#
#   ./downloadnet.sh vgg19
#   ./downloadnet.sh deepsim
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

target=$1

if [ "$target" = '' ]; then
    echo "Please specify the network ('vgg19' or 'deepsim') to be downloaded."
    exit 1
fi

# VGG19
if [ "$target" = 'vgg19' ]; then
    output=VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel
    srcurl=http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel

    [ -f $output ] && echo "$output already exists." && exit 0
    
    download_file $srcurl $output

    echo "$output saved."
    exit 0
fi

# Deepsim
if [ "$target" = 'deepsim' ]; then
    output=generator_for_inverting_fc7/generator.caffemodel
    srcurl=https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip

    [ -f $output ] && echo "$output already exists." && exit 0
    
    [ -f deepsim_v0.zip ] || download_file $srcurl deepsim_v0.zip
    [ -d deepsim_v0 ] || unzip deepsim_v0.zip
    [ -d $(dirname $output) ] || mkdir $(dirname $output)
    cp deepsim_v0/fc7/generator.caffemodel $output

    echo "$output saved."
    exit 0
fi

# Unknown target
echo "Unknown network $target"
exit 1
