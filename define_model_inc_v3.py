from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

def define_inc_v3_model_to_json(dest="Inceptionv3_CNN_CKplus_model.json", weights_init = None, nb_classes = 7, img_rows = 299, img_cols = 299, img_chs=3):
	if K.image_dim_ordering() == "th":
            base_model=InceptionV3(weights=weights_init, include_top=False, input_shape=(img_chs, img_rows, img_cols))
        else:
            base_model=InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3))
	# add a global spatial average pooling layer
	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	# let's add a fully-connected layer
	x = Dense(1024, activation='relu')(x)
	# and a logistic layer -- we have 7 classes
	predictions = Dense(nb_classes, activation='softmax')(x)
	# In[23]:

	model = Model(input=base_model.input, output=predictions)
	print model.summary()
	# serialize model to JSON
	model_json = model.to_json()
	with open(dest, "w") as json_file:
    	    json_file.write(model_json)
	print ("\nThe inc_V3 model is saved to a json file")
	return model
