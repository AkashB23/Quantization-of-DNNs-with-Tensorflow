import tensorflow as tf
output_node_names = "output node name"
input_node_names = "input node name"

input_array = [input_node_names]
output_array = [output_node_names]

converter = tf.lite.TFLiteConverter.from_frozen_graph('Frozen.pb',input_arrays=input_array,output_arrays=output_array)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("Output.tflite", "wb").write(tflite_quant_model)