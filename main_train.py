#Main function of the project

#train.train_network(augment_data=True, nb_epochs=200, batch_size=32, loss='categorical_crossentropy', optim = 'adam', X_train = np.load("X_train_True_299_299_3.npy"), X_test = np.load("X_test_True_299_299_3.npy"), Y_train = np.load("Y_train_True_299_299_3.npy"), Y_test = np.load("Y_test_True_299_299_3.npy"), model_from = "Inceptionv3_CNN_CKplus_model.json", logger=True, lr_reduce=True, min_lr = 0.0001, metrics = ['accuracy'])

import sys
import train_inc_v3 as train


def main(argv):
	if len(argv)>=3:
		optim = argv[2]
	else:
		optim = 'adam'
	train.train_network(nb_epochs = int(argv[0]), augment_data=(argv[1]=='augment'), optim=optim)
if __name__ == "__main__":
   main(sys.argv[1:])
