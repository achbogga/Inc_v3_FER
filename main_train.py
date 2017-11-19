#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  prepare_fer_2013.py
#  Code which prepares FER_2013 Kaggle challenge dataset from csv to .npy files which can be directly used with Inc_v3_FER repo for CNN training and testing.
#  For usage please type ./prepare_fer_2013.py --help
#  
#  Copyright 2017 Achyut Boggaram <achbogga[at]gmail[dot]com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys
from train_inc_v3 import train_network

from define_model_custom_cnn_for_fer import define_custom_cnn_model_to_json
import argparse
import numpy as np
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('nb_epochs', type=int, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, help='The number of samples per batch while training', default=256)
    parser.add_argument('--augment_data', type=bool, help='The flag whether to augment data while training', default=False)
    parser.add_argument('--define_model', type=bool, help='Flag whether to define a new inc_v3 model definition and save to the model file given or to load the model file instead', default=False)
    parser.add_argument('--custom_model', type=str, help='Flag whether to define a new inc_v3 model definition or define a custom_cnn', default='custom')
    parser.add_argument('--optimizer', type=str, help='The optimizer name to use while training', default='adam')
    parser.add_argument('--loss_function', type=str, help='The loss function name to use while training', default='categorical_crossentropy')
    parser.add_argument('image_size', type=int, help='Image size in pixels.')
    parser.add_argument('--image_channels', type=int, help='Desired image channels (ex: 3-RGB)', default=3)
    parser.add_argument('--nb_classes', type=int, help='The number of output classes', default=7)
    parser.add_argument('--validation_or_test', type=str, help='The test set to be used while training', default='validation')
    parser.add_argument('--model_file', type=str, help='The model definition file to be used for training', default='Inceptionv3_latest_model.json')
    parser.add_argument('--weights_file', type=str, help='The weights file to be loaded', default=None)
    return parser.parse_args(argv)

def main(args):
    info = '_'+str(args.image_size)+'_'+str(args.image_channels)+'.npy'
    X_train = np.load('npy/X_train'+info)
    Y_train = np.load('npy/Y_train'+info)
    X_test = np.load('npy/X_'+args.validation_or_test+info)
    Y_test = np.load('npy/Y_'+args.validation_or_test+info)
    print (args.define_model, args.custom_model)
    
    if args.define_model:
	if args.custom_model=='inc_v3':
	    from define_model_inc_v3 import define_inc_v3_model_to_json
	    define_inc_v3_model_to_json(args.model_file, args.weights_file, args.nb_classes, args.image_size, args.image_size, args.image_channels)
	elif args.custom_model=='custom':
	    define_custom_cnn_model_to_json(args.model_file, args.weights_file, args.nb_classes, args.image_size, args.image_size, args.image_channels)
    train_network(args.augment_data, args.nb_epochs, args.batch_size, args.loss_function, args.optimizer, X_train, X_test, Y_train, Y_test, args.model_file, logger=True, lr_reduce=True, min_lr = 0.0001, metrics = ['accuracy'])
if __name__ == "__main__":
   args = parse_arguments(sys.argv[1:])
   main(args)
