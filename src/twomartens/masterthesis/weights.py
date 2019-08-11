#  -*- coding: utf-8 -*-
#
#  Copyright 2019 Jim Martens
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Functionality to modify weights.

Functions:
    subsample_ssd(ssd_weights_file): sub-samples the weights of the SSD network
"""
import shutil

import h5py
import math
import numpy as np

from twomartens.masterthesis.ssd_keras.misc_utils import tensor_sampling_utils


def subsample_ssd(ssd_weights_file: str, subsampled_weights_file: str) -> None:
    """
    Sub-samples the weights of the SSD 300 network.
    
    Args:
        ssd_weights_file: file path for the SSD weights file
        subsampled_weights_file: file path for file with sub-sampled weights
    """
    shutil.copy(ssd_weights_file, subsampled_weights_file)
    with h5py.File(subsampled_weights_file) as weights_destination_file:
    
        classifier_names = ["conv4_3_norm_mbox_conf",
                            "fc7_mbox_conf",
                            "conv6_2_mbox_conf",
                            "conv7_2_mbox_conf",
                            "conv8_2_mbox_conf",
                            "conv9_2_mbox_conf"]
        n_classes_source = 81
        classes_of_interest = [i for i in range(61)]  # first 60 classes are kept plus background
        for name in classifier_names:
            # Get the trained weights for this layer from the source HDF5 weights file.
            kernel = weights_destination_file[name][name]['kernel:0'].value
            bias = weights_destination_file[name][name]['bias:0'].value
            
            # Get the shape of the kernel. We're interested in sub-sampling
            # the last dimension, 'o'.
            height, width, in_channels, out_channels = kernel.shape
            
            subsampling_indices = []
            for i in range(int(math.floor(out_channels / n_classes_source))):
                indices = np.array(classes_of_interest) + i * n_classes_source
                subsampling_indices.append(indices)
            subsampling_indices = list(np.concatenate(subsampling_indices))
            
            # Sub-sample the kernel and bias.
            # The `sample_tensors()` function used below provides extensive
            # documentation, so don't hesitate to read it if you want to know
            # what exactly is going on here.
            new_kernel, new_bias = tensor_sampling_utils.sample_tensors(
                weights_list=[kernel, bias],
                sampling_instructions=[height, width, in_channels, subsampling_indices],
                axes=[[3]],
                # The one bias dimension corresponds to the last kernel dimension.
                init=['gaussian', 'zeros'],
                mean=0.0,
                stddev=0.005
            )
            
            # Delete the old weights from the destination file.
            
            del weights_destination_file[name][name]['kernel:0']
            del weights_destination_file[name][name]['bias:0']
            # Create new data sets for the sub-sampled weights.
            weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
            weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)


if __name__ == "__main__":
    weights_file = "data/weights/ssd/VGG_coco_SSD_300x300_iter_400000.h5"
    weights_destination_file = "data/weights/ssd/VGG_coco_SSD_300x300_iter_400000_subsampled.h5"
    subsample_ssd(weights_file, weights_destination_file)
