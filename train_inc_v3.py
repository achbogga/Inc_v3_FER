# coding: utf-8
import numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, CSVLogger

X_train = np.load("X_train_True_299_299_3.npy")
print "X_training_shape: ",X_train.shape

X_test = np.load("X_test_True_299_299_3.npy")
print "X_testing_shape: ",X_test.shape

Y_train = np.load("Y_train_True_299_299_3.npy")
print "Y_training_shape: ",Y_train.shape

Y_test = np.load("Y_test_True_299_299_3.npy")
print "training_shape: ",Y_test.shape

# load json and create model
json_file = open("Inceptionv3_CNN_CKplus_model.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
#loaded_model.summary()
# load weights into new model
#loaded_model.load_weights("Inceptionv3_CNN_CKplus_model_1000_epochs.h5")
print("Loaded model")
#print (" and existing weights from disk")

model = loaded_model
nb_epochs = 20000
csv_logger = CSVLogger('training_from_scratch_'+'Inc_v3_'+str(nb_epochs)+'.log', separator=',', append=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', callbacks=[csv_logger, reduce_lr], metrics=['accuracy'])

# stop training when val_loss is no longer improving even after dynamic LR


model.fit(X_train, Y_train, callbacks=[csv_logger, reduce_lr], batch_size=32, nb_epoch=nb_epochs, verbose=1, validation_data=(X_test, Y_test))

loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
metrics_file = open('Inc_v3_CNN_CKplus_reload_v2_metrics.txt', 'a')
metrics_file.write("\nloss: "+str(float(loss))+"\n")
metrics_file.write("accuracy: "+str(float(accuracy)))
metrics_file.write("\nOptimizer, epochs: sgd_initlr=0.01, "+str(nb_epochs))
metrics_file.close()

# serialize weights to HDF5
model.save_weights("Inceptionv3_CNN_CKplus_model_" + str(nb_epochs) + "_epochs.h5")
print("Saved weights to disk\n")
