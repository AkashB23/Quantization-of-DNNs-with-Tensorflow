# Quantization-of-DNNs-with-Tensorflow
Includes all necessary files to arrive at a TFlite starting with checkpoints

#### Requirements\
1.Tensorflow 1.15.2 installed
2.python > 2.7

#### Refer the link https://github.com/tensorflow/models/tree/master/research/slim to know all the necessary file citations mentioned here.
1.Need to clone the repository with above link.\
2.Copy all the Files provided to the folder with slim clone.

#### Training any of the CNN models provided in the Tensorflow slim repository would provide you with Checkpoints that are source to use files in this repository.

##### 1.create_model.py
Provide the path to checkpoints and other input details as mentioned in the comments provided in the code to get `Frozen.pb` protocolbuffer file of the trained CNN model.
##### 2.WQuantization.py
Specify the input and output node along with path to previously generated `Frozen.pb` file, the `output.tflite` file is the Flatbuffer file which is 1/4th size of the Floating point model and can be used in Android apps and other embedded platforms supported by tensorflow (here quantization refers only to weight quantization).
##### 3.visualize.py
Inorder to Graphically visualize the trained CNN/DNN architecture you can use this file and get a logs folder created as output later, with the tensorboard preinstalled in the device. `tensorboard --logdir=#(path to logs folder)` will land you in a the tensoeflow graph.
