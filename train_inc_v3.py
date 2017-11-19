# coding: utf-8

import sys
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam, SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

def train_network(augment_data, nb_epochs, batch_size, loss, optim, X_train, X_test, Y_train, Y_test, model_from, logger=True, lr_reduce=True, min_lr = 0.0001, metrics = ['accuracy']):

#------------------------dataaugmentation---------------------------------------------------#
	print ('Augment Data: ', augment_data)
	print ('No. of epochs: ', nb_epochs)
	print ('Batch size: ', batch_size)
	
	if (augment_data):

		datagen = ImageDataGenerator(
			featurewise_center=True,
			featurewise_std_normalization=True,
			rotation_range=20,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

		# compute quantities required for featurewise normalization
		# (std, mean, and principal components if ZCA whitening is applied)
		datagen.fit(X_train)

		print "X_training_shape: ",X_train.shape
		print "X_testing_shape: ",X_test.shape
		print "Y_training_shape: ",Y_train.shape
		print "Y_testing_shape: ",Y_test.shape

#-----------------------------Model-----------------------------------------------------#

	# load json and create model
	json_file = open(model_from, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#loaded_model.summary()
	# load weights into new model
	#loaded_model.load_weights("Inceptionv3_CNN_CKplus_model_1000_epochs.h5")
	print("Loaded model")
	#print (" and existing weights from disk")
#------------------------------------callbacks--------------------------------------------#

	model = loaded_model
	#nb_epochs = int(argv[0])
	meta_data = model_from+'_'+str(nb_epochs)+'_'+str(X_train.shape[0])+'_'+str(X_train.shape[1])+'_'+str(X_train.shape[3])
	if logger:
		csv_logger = CSVLogger('log/training_from_scratch_'+meta_data+'.log', separator=',', append=False)
	if lr_reduce:
		reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=min_lr)

	callbacks = []
	if logger:
		callbacks.append(csv_logger)
	if lr_reduce:
		callbacks.append(reduce_lr)
#-----------------------------------model compilation------------------------------------#

	model.compile(optimizer=optim, loss=loss, metrics=metrics)

#-----------------------------Actual Training-------------------------------------------#
	if (augment_data):
		print "\nEpochs argument: ",(int(nb_epochs))

		model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), callbacks=callbacks, nb_epoch= int(nb_epochs), samples_per_epoch=len(X_train), validation_data=(X_test, Y_test))
	else:
		model.fit(X_train, Y_train, callbacks = callbacks , batch_size=batch_size, nb_epoch = nb_epochs, verbose=1, validation_data=(X_test, Y_test))

	loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
	metrics_file = open('metrics/'+meta_data+'_metrics.log', 'w')
	metrics_file.write("\nloss: "+str(float(loss))+"\n")
	metrics_file.write("accuracy: "+str(float(accuracy)))
	metrics_file.write("\nOptimizer, epochs: sgd_initlr=0.01, "+str(nb_epochs))
	metrics_file.close()

	# serialize weights to HDF5
	model.save_weights("h5/Inceptionv3_CNN_CKplus_model_" + str(nb_epochs) + "_epochs.h5")
	print("Saved weights to disk\n")
