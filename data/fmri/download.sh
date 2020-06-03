#!/bin/bash

set -Ceu

[ -e sub-01_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O sub-01_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830643
[ -e sub-02_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O sub-02_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830712
[ -e sub-03_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O sub-03_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830862
[ -e sub-01_perceptionNaturalImageTest_original_VC.h5 ] || wget -O sub-01_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830631
[ -e sub-02_perceptionNaturalImageTest_original_VC.h5 ] || wget -O sub-02_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830697
[ -e sub-03_perceptionNaturalImageTest_original_VC.h5 ] || wget -O sub-03_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830856
