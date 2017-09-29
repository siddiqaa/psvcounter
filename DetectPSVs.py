from subprocess import call
import sys
import os
from PIL import Image
import numpy as np
from scipy import ndimage
from subprocess import call
import sys
import os
from PIL import Image
from datetime import datetime
import cv2

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

# intiate tensorflow and load custom model
import tensorflow as tf
import numpy as np



CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '<CONFIGURE_YOUR_PATH_HERE>_output_inference_graph_21.pb/frozen_inference_graph.pb'  

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '<CONFIGURE_YOUR_PATH_HERE>e/data/label_map.pbtxt'

NUM_CLASSES = 5

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

# the 0.8 is the confidence threshold. This script sorts the detected classes in descending confidence level and the code below checks to see if any of the top five detected objects match our target class with a threshold confidence greater than 80%
    if ((np.squeeze(scores)[0] > 0.8) and (np.squeeze(classes)[0] == 1)) \
            or ((np.squeeze(scores)[1] > 0.4) and (np.squeeze(classes)[1] == 1)) \
            or ((np.squeeze(scores)[2] > 0.4) and (np.squeeze(classes)[2] == 1)) \
            or ((np.squeeze(scores)[3] > 0.4) and (np.squeeze(classes)[3] == 1)) \
            or ((np.squeeze(scores)[4] > 0.4) and (np.squeeze(classes)[4] == 1)):
        print(str(np.squeeze(scores)[0]) + ' ' + str(segmentIndex))

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)


        img = Image.fromarray(image_np, 'RGB')
        #print(os.path.splitext(segmentFileName)[0])
        img.save("<CONFIGURE_PATH_TO_SAVING_THE_IMAGE_SEGMENTS_WITH_BOUNDINGBOXES> +"_detected.jpg")
        #img.show()

pageHeight = 3300
pageWidth = 5100

cropHeight = int(pageHeight/30)
cropWidth = int(pageWidth/50)

pageIndex = 1

with tf.Session(graph=detection_graph) as sess:
    startTime = datetime.now()





        imageNumpyArray = ndimage.imread("<CONFIGURE_PATH_TO_JPG_FILE_TO_CONDUCT_OBJECT_DETECTION_ON>")
 
        overlapWidth = 10

        segmentIndex = 0
        while segmentIndex <= 1499:
             if (segmentIndex == 0):
                cropArray = imageNumpyArray[0:cropHeight+overlapWidth,0:cropWidth+overlapWidth,:]
 
            # catch top right corner tile
            elif (segmentIndex == 49):
                cropArray = imageNumpyArray[0:cropHeight+overlapWidth,segmentIndex*cropWidth-overlapWidth:segmentIndex*cropWidth+cropWidth,:]
            
# catch bottom left corner tile
            elif (segmentIndex == 1450):
                cropArray = imageNumpyArray[cropHeight*segmentIndex//50-overlapWidth:cropHeight*segmentIndex//50+cropHeight,0:cropWidth+overlapWidth,:] 
            # catch bottom right corner tile
            elif (segmentIndex == 1499):
                cropArray = imageNumpyArray[cropHeight*segmentIndex//50-cropHeight-overlapWidth:cropHeight*segmentIndex//50,segmentIndex%50*cropWidth-overlapWidth:segmentIndex%50*cropWidth+cropWidth,:]
              #catch right edge tiles so no overlap on left
            elif (segmentIndex % 50 == 0):
                #print(cropHeight*segmentIndex//50)
                cropArray = imageNumpyArray[cropHeight*(segmentIndex//50):cropHeight*(segmentIndex//50)+cropHeight,0:cropWidth+overlapWidth,:]
                #print(cropArray.shape)
                #cropImage = Image.fromarray(cropArray, "RGB")
                #cropImage.save(directoryName + "/segments/" + pageFileBaseName + "/"+ pageFileBaseName + "_" + str(segmentIndex) + ".jpg")
            #catch top edge tiles so no overlap on top
            elif (segmentIndex <= 48):
                #print(segmentIndex*cropWidth)
                cropArray = imageNumpyArray[0:cropHeight + overlapWidth,segmentIndex*cropWidth:segmentIndex*cropWidth+cropWidth,:]
                #print(cropArray.shape)
                #cropImage = Image.fromarray(cropArray, "RGB")
                #cropImage.save(directoryName + "/segments/" + pageFileBaseName + "/"+ pageFileBaseName + "_" + str(segmentIndex) + ".jpg")
            # catch left edge tiles so no overlap on left
            elif (segmentIndex+1)%50 == 0:
               # print(segmentIndex * cropWidth)
                cropArray = imageNumpyArray[((segmentIndex+1)//50)*cropHeight-overlapWidth:((segmentIndex+1)//50)*cropHeight + cropHeight + overlapWidth,
                            (segmentIndex)%50 * cropWidth - overlapWidth:(segmentIndex)%50 * cropWidth + cropWidth, :]
                #print(cropArray.shape)
                #cropImage = Image.fromarray(cropArray, "RGB")
                #cropImage.save(directoryName + "/segments/" + pageFileBaseName + "/" + pageFileBaseName + "_" + str(segmentIndex) + ".jpg")
            # catch bottom edge tiles so no overlap on top
            elif (segmentIndex > 1450):
                #print(segmentIndex * cropWidth)
                cropArray = imageNumpyArray[((segmentIndex+1)//50)*cropHeight:((segmentIndex+1)//50)*cropHeight + cropHeight + overlapWidth,
                            (segmentIndex)%50 * cropWidth - overlapWidth:(segmentIndex)%50 * cropWidth + cropWidth, :]
                #print(cropArray.shape)
                #cropImage = Image.fromarray(cropArray, "RGB")
                #cropImage.save(directoryName + "/segments/" + pageFileBaseName + "/" + pageFileBaseName + "_" + str(segmentIndex) + ".jpg")
            else:
                cropArray = imageNumpyArray[(segmentIndex // 50) * cropHeight - overlapWidth: (segmentIndex // 50) * cropHeight + cropHeight + overlapWidth,(segmentIndex) % 50 * cropWidth - overlapWidth:(segmentIndex) % 50 * cropWidth + cropWidth + overlapWidth, :]
                #print(cropArray.shape)
                #cropImage = Image.fromarray(cropArray, "RGB")
                #cropImage.save(directoryName + "/segments/" + pageFileBaseName + "/" + pageFileBaseName + "_" + str(segmentIndex) + ".jpg")
            detect_objects(cropArray, sess, detection_graph)
            if segmentIndex%150 == 0:
                print(str(segmentIndex//150 * 10) + " percent complete")
            segmentIndex += 1

        totalTime = datetime.now() - startTime
        averageTimePerPage = totalTime / pageIndex

        print("Average time per page is " + str(averageTimePerPage) + ". Time remaining is " + averageTimePerPage * (len(pageFileNames) - pageIndex))
        pageIndex += 1
