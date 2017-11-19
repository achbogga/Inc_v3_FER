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

import os
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
from keras.utils import np_utils

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('fer_csv', type=str, help='The dataset csv file after extracting the FER 2013 dataset')
    parser.add_argument('output_dir', type=str, help='The dataset output directory to save prepared .npy files')
    parser.add_argument('--align', type=int, help='Flag to align the faces usign mtcnn alignemnt 0-False, 1-True ', default=0)
    parser.add_argument('--image_size', type=int, help='Image size (height, width) in pixels.', default=48)
    parser.add_argument('--image_channels', type=int, help='Desired image channels (ex: 3-RGB)', default=3)   
    return parser.parse_args(argv)

def save_arrays(X, Y, name, args):
	X = np.asarray(X, dtype=np.float16)
	Y = np.asarray(Y, dtype=np.uint8)
	nb_classes = len(set(Y))
	print (name+'_no_of_classes_detected: ', nb_classes)
	Y = np_utils.to_categorical(Y, nb_classes)
	X /= 255.0
	info = str(args.image_size)+'_'+str(args.image_channels)
	print (name, X.shape, Y.shape)
	np.save('npy/X_'+name+'_'+info+'.npy', X)
	np.save('npy/Y_'+name+'_'+info+'.npy', Y)
	

def prepare_fer(args):
	training_images = []
	training_labels = []
	public_test_images = []
	public_test_labels = []
	private_test_images = []
	private_test_labels = []
	with open(args.fer_csv, 'rb') as csvfile:
		csvreader = csv.DictReader(csvfile, delimiter=',')
		for row in csvreader:
			if row['Usage']=='Training':
				training_labels.append(int(row['emotion']))
				single_channel_image = np.asarray(row['pixels'].split(' '), dtype=np.int16).reshape((48, 48))
				img = single_channel_image
				img = zoom(img, float(args.image_size)/48.0)
				training_images.append(np.stack((img,)*args.image_channels, axis=-1))
			elif row['Usage']=='PublicTest':
				public_test_labels.append(int(row['emotion']))
				single_channel_image = np.asarray(row['pixels'].split(' '), dtype=np.int16).reshape((48, 48))
				img = single_channel_image
				img = zoom(img, float(args.image_size)/48.0)
				public_test_images.append(np.stack((img,)*args.image_channels, axis=-1))
			elif row['Usage']=='PrivateTest':
				private_test_labels.append(int(row['emotion']))
				single_channel_image = np.asarray(row['pixels'].split(' '), dtype=np.int16).reshape((48, 48))
				img = single_channel_image
				img = zoom(img, float(args.image_size)/48.0)
				private_test_images.append(np.stack((img,)*args.image_channels, axis=-1))
	save_arrays(training_images, training_labels, 'train', args)
	save_arrays(public_test_images, public_test_labels, 'validation', args)
	save_arrays(private_test_images, private_test_labels, 'test', args)
	return 0

if __name__ == '__main__':
    import sys
    args = parse_arguments(sys.argv[1:])
    sys.exit(prepare_fer(args))
