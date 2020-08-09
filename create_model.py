import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
import os
from nets import nets_factory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def create_model(logs_path, model_name):
    print("Writing model files.")
    tf.reset_default_graph()
    chk_name = os.path.join(logs_path, model_name)
    network_fn = nets_factory.get_network_fn( 'mobilenet_v2', num_classes=1001) #model name from the listed ones in the nets_factory in slim repo
    inputs = tf.placeholder(tf.float32, shape = (1,224,224,3), name = 'input') # shape is the shape  of the input
    logits, end_points = network_fn(inputs)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, chk_name)
    graph = tf.get_default_graph()
    """for op in graph.get_operations():
        print (op.name)"""
    input_node_names = "input" # input node name pf the CNN
    output_node_names = "MobilenetV2/Predictions/Reshape_1" # output node name of the CNN, refer the graph file generated to know input and output.
    output_graph_def = graph_util.convert_variables_to_constants(sess,
        graph.as_graph_def(), output_node_names.split(","))
    out_path = "."
    graph_io.write_graph(output_graph_def, out_path,'FrozenGraph.pb', as_text=False) #Output file Frozen graph of the model
    

def main():
    logs_path =  'path to folder containing checkpoints' 
    model_name =  "model.ckpt"    #checkpoint file name.
    create_model(logs_path, model_name)

if __name__ == "__main__":
    main()