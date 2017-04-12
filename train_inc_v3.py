# coding: utf-8

import sys
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam, SGD,
from keras.callbacks import ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

def train_network(augment_data=True, nb_epochs=200, batch_size=32, loss='categorical_crossentropy', optim = 'adam', X_train = np.load("npy/X_train_True_299_299_3.npy"), X_test = np.load("npy/X_test_True_299_299_3.npy"), Y_train = np.load("npy/Y_train_True_299_299_3.npy"), Y_test = np.load("npy/Y_test_True_299_299_3.npy"), model_from = "Inceptionv3_CNN_CKplus_model.json", logger=True, lr_reduce=True, min_lr = 0.0001, metrics = ['accuracy']):

#------------------------dataaugmentation---------------------------------------------------#

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
	if logger:
		csv_logger = CSVLogger('log/training_from_scratch_'+'Inc_v3_'+str(nb_epochs)+'.log', separator=',', append=False)
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
		model.fit(X_train, Y_train, callbacks = callbacks , batch_size=batch_size, nb_epoch = nb_epochs, verbose=1, validation_data=(X_test, Y_test), metrics=metrics)

	loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
	metrics_file = open('metrics/Inc_v3_CNN_CKplus_reload_v_data_augmented_metrics.txt', 'a')
	metrics_file.write("\nloss: "+str(float(loss))+"\n")
	metrics_file.write("accuracy: "+str(float(accuracy)))
	metrics_file.write("\nOptimizer, epochs: sgd_initlr=0.01, "+str(nb_epochs))
	metrics_file.close()

	# serialize weights to HDF5
	model.save_weights("h5/Inceptionv3_CNN_CKplus_model_" + str(nb_epochs) + "_epochs.h5")
	print("Saved weights to disk\n")
