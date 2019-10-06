import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #pass
		# load frozen tensorflow model into memory
        GRAPH_FILE = 'light_classification/model/frozen_inference_graph_sim.pb'
        self.detection_graph = load_graph(GRAPH_FILE)
        # create tensorflow session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(graph=self.detection_graph, config=config)
        
		# `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        # extract the relevant tensors which reflect the outputs of the graph
		ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        self.tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                self.tensor_dict[key] = self.detection_graph.get_tensor_by_name(
                tensor_name)
		
		# extract the relevant tensors which reflect the input of the graph
		# The input placeholder for the image.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return TrafficLight.UNKNOWN

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def run_inference_for_single_image(self, image, graph):
        """Run detection and classification on a sample image.

        Args:
            image：
            graph：

        Returns:
        """
        output_dict = self.session.run(self.tensor_dict,
                                feed_dict={self.image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        return output_dict