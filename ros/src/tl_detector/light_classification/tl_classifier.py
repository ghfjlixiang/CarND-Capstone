import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        #pass
		self.is_site = FALSE
        self.image_result_show = False
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
        light_thresh = 0.4
        current_light = TrafficLight.UNKNOWN
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_cp = image.copy()
        image_np_expanded = np.expand_dims(image_cp, axis=0)

        # Actual detection.
        output_dict = self.run_inference_for_single_image(image_np_expanded, self.detection_graph)

        # Show classified result image for test
        if self.image_result_show:
            img_width,img_height,depth = image.shape
            font = cv.FONT_HERSHEY_COMPLEX
            category_index = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'},
                    3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}
            for i in range(output_dict['num_detections']):
                top_y = output_dict['detection_boxes'][i][0] * img_width
                top_x = output_dict['detection_boxes'][i][1] * img_height
                botom_y = output_dict['detection_boxes'][i][2] * img_width
                botom_x = output_dict['detection_boxes'][i][3] * img_height
                score = output_dict['detection_scores'][i]
                class_name = category_index[output_dict['detection_classes'][i]]['name']
                if score > light_thresh:
                    cv.rectangle(image_cp, (int(top_x), int(top_y)), (int(botom_x), int(botom_y)), (0, 0, 255), thickness=2)
                    cv.putText(image_cp, str(class_name) + ' ' + str(score), (int(top_x), int(top_y)), font, 0.8,(255, 255, 255), 1)
            cv.namedWindow('result', cv.WINDOW_NORMAL)
            cv.imshow('result', image_cp)
            cv.waitKey(1)
        
        # init red light score
        red_score_max = 0
        img_width,img_height,depth = image.shape
        for i in range(output_dict['num_detections']):
            if self.is_site or output_dict['detection_classes'][i]==2:        #if sim, just get red light max score 
                if output_dict['detection_scores'][i] > red_score_max:        #if site, need get all traffic light max score
                    red_score_max = output_dict['detection_scores'][i]
                    top_y = output_dict['detection_boxes'][i][0] * img_width
                    top_x = output_dict['detection_boxes'][i][1] * img_height
                    botom_y = output_dict['detection_boxes'][i][2] * img_width
                    botom_x = output_dict['detection_boxes'][i][3] * img_height
        if self.is_site:       # if site mode, need classified the light color based image color
            if red_score_max > light_thresh:
                lightImg = image[int(top_y):int(botom_y),int(top_x):int(botom_x)]   # crop traffic image
                current_light = self.light_color(lightImg)        # get traffic light color
            else:
                current_light = TrafficLight.GREEN
        else:
            if red_score_max > light_thresh:
                current_light = TrafficLight.RED
            else:
                current_light = TrafficLight.GREEN 

        if current_light == TrafficLight.RED:
            print('currend light is RED')
        else:
            print('currend light is GREEN')
		
        return current_light

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

    def light_color(self,lightImg):
        """.

        Args:
            lightImg：

        Returns:
        """        
        if self.image_result_show:
            cv.namedWindow('light_image', cv.WINDOW_NORMAL)
            cv.imshow('light_image', lightImg)
            cv.waitKey(1)

        lightImg = cv.cvtColor(lightImg, cv.COLOR_BGR2HSV)    # translate bgr to HSV
        
        height,width,depth = lightImg.shape 
    
        height_start = int (height/8)      # split 3 part in height 
        height_1st = int(height/3)        
        height_2nd = int(height*2/3)      
        height_end = int(height*7/8)

        lightImg_red = lightImg[height_start:height_1st,:,2]     # red part image
        lightImg_yellow = lightImg[height_1st:height_2nd,:,2]    # yellow part
        lightImg_green = lightImg[height_2nd:height_end,:,2]     # green part

        # Calculating the average of pixels each color part image
        red_height,red_width = lightImg_red.shape
        red_pixel_sum = np.sum(lightImg_red)/(red_height*red_width)

        yellow_height,yellow_width = lightImg_yellow.shape
        yellow_pixel_sum = np.sum(lightImg_yellow)/(yellow_height*yellow_width)

        green_height,green_width = lightImg_green.shape
        green_pixel_sum = np.sum(lightImg_green)/(green_height*green_width)

        print ("red is%f, yellow is %f, green is %f"%(red_pixel_sum,yellow_pixel_sum,green_pixel_sum))
        # red part is biggest indicate red light
        if red_pixel_sum > yellow_pixel_sum and red_pixel_sum > green_pixel_sum:
            current_light = TrafficLight.RED
        else:
            current_light = TrafficLight.GREEN
        return current_light