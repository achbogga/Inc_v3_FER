from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from custom_cnn import custom_cnn
from keras import backend as K

def define_custom_cnn_model_to_json(dest, weights_init, nb_classes, img_rows, img_cols, img_chs):
	if K.image_dim_ordering() == "th":
            base_model=custom_cnn(weights=weights_init, include_top=False, input_shape=(img_chs, img_rows, img_cols))
        else:
            base_model=custom_cnn(weights=None, include_top=False, input_shape=(img_rows, img_cols, img_chs))
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(256, activation='relu')(x)
	# and a logistic layer -- we have 7 classes
	predictions = Dense(nb_classes, activation='softmax')(x)
	# In[23]:

	model = Model(input=base_model.input, output=predictions)
	print model.summary()
	# serialize model to JSON
	model_json = model.to_json()
	with open(dest, "w") as json_file:
    	    json_file.write(model_json)
	print ("\nThe custom cnn model is saved to a json file")
	return model
