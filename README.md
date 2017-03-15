# Inc_v3_FER
Facial Expression Recognition using Inception V3 Model from keras
This is a baseline to my thesis on Facial Expression recognition with videos.
Please feel free to email me at achbogga@gmail.com if you have any questions about this project.
Documentation of this project is in progress and will be updated constantly.

Details:
Datasets used: CK+ (Cohn Kanade Extended) 327 static labeled images with facial expressions.
Model used: Inception V3 provided by keras as an application. (Input shape: 299,299,3 if TF backend is used.)
Face Registration used: OpenCV Haar Cascades multi-scale classifier face detector

Outline:
Data preperation:
  load_data.py
  functions:
    copy_to_n_channels(input_arr, n): -> returns nd array
    convert_to_grey_scale(source, dest, img_rows, img_cols): -> returns nd array
    prepare_for_inc_v3(img_rows=299, img_cols=299, img_chs=3, out_put_classes=7, img_src='', label_src='', asGray=False,           face_detection_xml ="") -> saves the X_train, X_test, Y_train, Y_test files as .npy files for further usage.
Model Definition:
 define_model_inc_v3.py
  functions:
    define_inc_v3_model_to_json(dest="", weights_init = None, nb_classes = 7, img_rows = 299, img_cols = 299, img_chs=3):
    -> defines model directly from keras applications and initializes with or without precious weights
    -> saves the model to a .json file for further usage
Training:
  train_inc_v3.py
  -> contains main which takes first command line argument as the number of epochs/iterations to train the model.
  -> Logs the training metrics in a .log file named after the number of iterations
  -> saves the weights after completion to disk similarly
  
Any suggestions are welcome. This is still a work in progress. Thanks!
