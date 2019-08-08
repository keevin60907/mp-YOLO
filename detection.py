'''
Object Detection on Panorama pictures
Usage:
    $ pyhton3 detection.py <pano_picture> <output_picture>

    pano_picture(str)  : the pano pic file
    output_picture(str): the result picture
'''
import sys
import cv2
import numpy as np
from stereo import pano2stereo, realign_bbox

CF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_RESOLUTION = (416, 416)

class Yolo():
    '''
    Packed yolo Netwrok from cv2
    '''
    def __init__(self):
        # get model configuration and weight
        model_configuration = 'yolov3.cfg'
        model_weight = 'yolov3.weights'

        # define classes
        self.classes = None
        class_file = 'coco.names'
        with open(class_file, 'rt') as file:
            self.classes = file.read().rstrip('\n').split('\n')

        net = cv2.dnn.readNetFromDarknet(
            model_configuration, model_weight)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.yolo = net

        self.cf_th = CF_THRESHOLD
        self.nms_th = NMS_THRESHOLD
        self.resolution = INPUT_RESOLUTION
        print('Model Initialization Done!')

    def detect(self, frame):
        '''
        The yolo function which is provided by opencv

        Args:
            frames(np.array): input picture for object detection

        Returns:
            ret(np.array): all possible boxes with dim = (N, classes+5)
        '''
        blob = cv2.dnn.blobFromImage(np.float32(frame), 1/255, self.resolution,
                                     [0, 0, 0], 1, crop=False)

        self.yolo.setInput(blob)
        layers_names = self.yolo.getLayerNames()
        output_layer =\
            [layers_names[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        outputs = self.yolo.forward(output_layer)

        ret = np.zeros((1, len(self.classes)+5))
        for out in outputs:
            ret = np.concatenate((ret, out), axis=0)
        return ret

    def draw_bbox(self, frame, class_id, conf, left, top, right, bottom):
        '''
        Drew a Bounding Box

        Args:
            frame(np.array): the base image for painting box on
            class_id(int)  : id of the object
            conf(float)    : confidential score for the object
            left(int)      : the left pixel for the box
            top(int)       : the top pixel for the box
            right(int)     : the right pixel for the box
            bottom(int)    : the bottom pixel for the box

        Return:
            frame(np.array): the image with bounding box on it
        '''
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(class_id < len(self.classes))
            label = '%s:%s' % (self.classes[class_id], label)

        #Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame,
                      (left, top - round(1.5*label_size[1])),
                      (left + round(label_size[0]), top + base_line),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        return frame

    def nms_selection(self, frame, output):
        '''
        Packing the openCV Non-Maximum Suppression Selection Algorthim

        Args:
            frame(np.array) : the input image for getting the size
            output(np.array): scores from yolo, and transform into confidence and class

        Returns:
            class_ids (list)  : the list of class id for the output from yolo
            confidences (list): the list of confidence for the output from yolo
            boxes (list)      : the list of box coordinate for the output from yolo
            indices (list)    : the list of box after NMS selection

        '''
        print('NMS selecting...')
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class
        # with the highest score.
        class_ids = []
        confidences = []
        boxes = []
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CF_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CF_THRESHOLD, NMS_THRESHOLD)

        return class_ids, confidences, boxes, indices

    def process_output(self, input_img, frames):
        '''
        Main progress in the class.
        Detecting the pics >> Calculate Re-align BBox >> NMS selection >> Draw BBox

        Args:
            input_img(np.array): the original pano image
            frames(list)       : the results from pan2stereo, the list contain four spects of view

        Returns:
            base_frame(np.array): the input pano image with BBoxes
        '''
        height = frames[0].shape[0]
        width = frames[0].shape[1]
        first_flag = True
        outputs = None

        print('Yolo Detecting...')
        for face, frame in enumerate(frames):
            output = self.detect(frame)
            for i in range(output.shape[0]):
                output[i, 0], output[i, 1], output[i, 2], output[i, 3] =\
                realign_bbox(output[i, 0], output[i, 1], output[i, 2], output[i, 3], face)
            if not first_flag:
                outputs = np.concatenate([outputs, output], axis=0)
            else:
                outputs = output
                first_flag = False

        base_frame = input_img
        # need to inverse preoject
        class_ids, confidences, boxes, indices = self.nms_selection(base_frame, outputs)
        print('Painting Bounding Boxes..')
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.draw_bbox(base_frame, class_ids[i], confidences[i],
                           left, top, left + width, top + height)

        return base_frame

def main():
    '''
    For testing now..
    '''
    my_net = Yolo()

    input_pano = cv2.imread(sys.argv[1])
    projections = pano2stereo(input_pano)

    output_frame = my_net.process_output(input_pano, projections)
    cv2.imwrite(sys.argv[2], output_frame)

if __name__ == '__main__':
    main()
