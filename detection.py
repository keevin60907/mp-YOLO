import cv2
import sys
import numpy as np
import os.path
from stereo import pano2stereo, stereo2pano

CF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_RESOLUTION = (608, 608)

class yolo():
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
        with open(class_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        net = cv2.dnn.readNetFromDarknet(
            model_configuration, model_weight)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.yolo = net

        self.cf_th = CF_THRESHOLD = 0.5
        self.nms_th = NMS_THRESHOLD = 0.4
        self.resolution = INPUT_RESOLUTION
        print('Model Initialization Done!')

    def detect(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1/255, 
            self.resolution, [0, 0, 0], 1, crop=False)
        
        self.yolo.setInput(blob)
        layers_names = self.yolo.getLayerNames()
        output_layer =\
            [layers_names[i[0] - 1] for i in self.yolo.getUnconnectedOutLayers()]
        outputs = self.yolo.forward(output_layer)
        
        ret = np.zeros((1, len(self.classes)+5))
        for out in outputs:
            ret = np.concatenate((ret, out), axis=0)
        return ret

    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
        label = '%.2f' % conf
        
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
        return frame

    def NMS_selection(self, frame, output):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for detection in output:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > CF_THRESHOLD:
                center_x = int(detection[0] * frame_width / 2)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width / 2)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CF_THRESHOLD, NMS_THRESHOLD)

        return classIds, confidences, boxes, indices

    def process_output(self, frame_0, frame_1, frame_2, frame_3):
        height = frame_0.shape[0]
        width = frame_0.shape[1]

        output_0 = self.detect(frame_0)
        for i in range(output_0.shape[0]):
            output_0[i, 0] += 1/2
        output_1 = self.detect(frame_1)
        for i in range(output_1.shape[0]):
            output_0[i, 0] += 1
        output_2 = self.detect(frame_2)
        for i in range(output_2.shape[0]):
            output_0[i, 0] += 3/2
        output_3 = self.detect(frame_3)

        output = np.concatenate((output_0, output_1, output_2, output_3), axis=0)
        output[output > 2] -= 2

        # need to inverse preoject
        output_frame = np.concatenate([frame_3, frame_1], axis=1)
        classIds, confidences, boxes, indices = self.NMS_selection(output_frame, output)
        for i in indices:
            print(i)
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(output_frame, classIds[i], confidences[i], left, top, left + width, top + height)

        return output_frame



if __name__ == '__main__':
    myNet = yolo()
    frame_0 = cv2.imread('./projection example/face_0.jpg')
    frame_1 = cv2.imread('./projection example/face_1.jpg')
    frame_2 = cv2.imread('./projection example/face_2.jpg')
    frame_3 = cv2.imread('./projection example/face_3.jpg')

    output_frame = myNet.process_output(frame_0, frame_1, frame_2, frame_3)
    cv2.imwrite('./result.jpg', output_frame)
