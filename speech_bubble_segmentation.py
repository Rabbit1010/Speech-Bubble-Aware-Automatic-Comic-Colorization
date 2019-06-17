# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 20:56:00 2019

@author: Wei-Hsiang, Shen
"""

# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import os
import cv2


def Detect_Text_with_EAST(net, image):
    """Detect the text inside an image with EAST and return its bounding boxes"""
    CONFIDENCE_THRESHOLD = 0.5

    (H, W) = image.shape[:2]

    # resize input resolution to match the 1st input layer of EAST
    (newW, newH) = (320, 320)
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [ "feature_fusion/Conv_7/Sigmoid",	"feature_fusion/concat_3"]

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
#    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
#    end = time.time()

    # show timing information on text prediction
#    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
    	# extract the scores (probabilities), followed by the geometrical
    	# data used to derive potential bounding box coordinates that
    	# surround text
    	scoresData = scores[0, 0, y]
    	xData0 = geometry[0, 0, y]
    	xData1 = geometry[0, 1, y]
    	xData2 = geometry[0, 2, y]
    	xData3 = geometry[0, 3, y]
    	anglesData = geometry[0, 4, y]

    	# loop over the number of columns
    	for x in range(0, numCols):
    		# if our score does not have sufficient probability, ignore it
    		if scoresData[x] < CONFIDENCE_THRESHOLD:
    			continue

    		# compute the offset factor as our resulting feature maps will
    		# be 4x smaller than the input image
    		(offsetX, offsetY) = (x * 4.0, y * 4.0)

    		# extract the rotation angle for the prediction and then
    		# compute the sin and cosine
    		angle = anglesData[x]
    		cos = np.cos(angle)
    		sin = np.sin(angle)

    		# use the geometry volume to derive the width and height of
    		# the bounding box
    		h = xData0[x] + xData2[x]
    		w = xData1[x] + xData3[x]

    		# compute both the starting and ending (x, y)-coordinates for
    		# the text prediction bounding box
    		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
    		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
    		startX = int(endX - w)
    		startY = int(endY - h)

    		# add the bounding box coordinates and probability score to
    		# our respective lists
    		rects.append((startX, startY, endX, endY))
    		confidences.append(scoresData[x])

    # non-maxima suppressio
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # Correct the bounding box coordinates to the origianl resolution
    boxes[:,0] = boxes[:,0]*rW
    boxes[:,1] = boxes[:,1]*rH
    boxes[:,2] = boxes[:,2]*rW
    boxes[:,3] = boxes[:,3]*rH

    # Calculate the center position of each bounding box
    boxes_center = np.zeros((boxes.shape[0],2))
    boxes_center[:,0] = (boxes[:,0]+boxes[:,2])//2 # center x position
    boxes_center[:,1] = (boxes[:,1]+boxes[:,3])//2 # center y position

    # Eliminate lonely boxes
    MIN_DIST = int(150*H/809)
    valid_list = []
    for (x_pos, y_pos) in boxes_center: # for each point
        # calculate the distance to each other points
        dist = np.sqrt((boxes_center[:,0]-x_pos)**2 + (boxes_center[:,1]-y_pos)**2)
        if len(dist[dist<=MIN_DIST]) > 1:
            valid_list.append(True)
        else:
            valid_list.append(False)
    boxes = boxes[valid_list]
    boxes_center = boxes_center[valid_list]

    # Cluster the boxes
    group = np.zeros(boxes_center.shape[0])
    link_graph = np.zeros((boxes_center.shape[0], boxes_center.shape[0]))

    for i_box in range(len(group)):
        dist = np.sqrt((boxes_center[:,0]-boxes_center[i_box,0])**2 + (boxes_center[:,1]-boxes_center[i_box,1])**2)
        link_graph[i_box, :] = dist<=MIN_DIST

    while 0 in group:
        group[np.argmin(group)] = np.max(group) + 1
        for _ in range(len(group)):
            for i_box in range(len(group)):
                for i_neighbor in range(len(group)):
                    if link_graph[i_box, i_neighbor]==True: # there's a link
                        if group[i_box]!=0 and group[i_neighbor]==0:
                            group[i_neighbor] = group[i_box] # cluster the neighbor
                        elif group[i_box]==0 and group[i_neighbor]!=0:
                            group[i_box] = group[i_neighbor]

    # Merge the bounding boxes to a bigger bounxing box
    boxes_big = np.zeros((int(np.max(group)),4), dtype='int32')
    for i_group in range(1, int(np.max(group)+1)):
        x_start = np.min(boxes[group==i_group,0])
        y_start = np.min(boxes[group==i_group,1])
        x_end   = np.max(boxes[group==i_group,2])
        y_end   = np.max(boxes[group==i_group,3])

        # Make the bounding boxes slightly larger
        X_RATIO = 0.0
        Y_RATIO = 0.0
        width = x_end - x_start
        height = y_end - y_start
        x_start = int(x_start - width*X_RATIO)
        y_start = int(y_start - height*Y_RATIO)
        x_end = int(x_end + width*X_RATIO)
        y_end = int(y_end + height*Y_RATIO)

        # Prevent overflow
        if x_start<0: x_start = 0
        if y_start<0: y_start = 0
        if x_end>image.shape[1]*rW: x_end = image.shape[1]*rW
        if y_end>image.shape[0]*rH: y_end = image.shape[0]*rH


        boxes_big[i_group-1,0] = x_start
        boxes_big[i_group-1,1] = y_start
        boxes_big[i_group-1,2] = x_end
        boxes_big[i_group-1,3] = y_end

    return boxes, boxes_big


def Speech_Bubble_segmentation(img_path=None, image=None):
    if img_path!=None:
        image = cv2.imread(img_path)
    net_EAST = cv2.dnn.readNet('./checkpoints/frozen_east_text_detection.pb')
    boxes, boxes_big = Detect_Text_with_EAST(net_EAST, image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bubble_flood_fill = np.zeros((image.shape[0], image.shape[1]))
    # Magic picker
    for (x_start, y_start, x_end, y_end) in boxes_big:
        start_x = (x_start+x_end)//2
        start_y = y_start

        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        out = cv2.floodFill(image, mask, (start_x, start_y), 255, (10,10,10), (10,10,10));
        bubble_flood_fill = np.logical_xor(bubble_flood_fill, mask[1:-1, 1:-1]*255)

    # Hold Filling
    bubble_segmentation = np.array(bubble_flood_fill, dtype='uint8')
    for (x_start, y_start, x_end, y_end) in boxes:
        bubble_segmentation[y_start:y_end, x_start:x_end] = 1
    bubble_segmentation *= 255
    kernel = np.ones((20,20),np.uint8)
    bubble_segmentation = np.pad(bubble_segmentation, 40, mode='constant')
    bubble_segmentation = cv2.morphologyEx(bubble_segmentation, cv2.MORPH_CLOSE, kernel)
    bubble_segmentation = bubble_segmentation[40:-40, 40:-40]

    # Show result
    image = cv2.imread(img_path)
    for (startX, startY, endX, endY) in boxes:
    	# draw the bounding box on the image
    	cv2.rectangle(image, (startX, startY), (endX, endY), (66, 255, 255), 3)

    for (startX, startY, endX, endY) in boxes_big:
    	# draw the bounding box on the image
    	cv2.rectangle(image, (startX, startY), (endX, endY), (100, 255, 100), 3)

    # show the output image
#    cv2.imshow("Text Detection", image)
#    cv2.waitKey(0)
#
#    cv2.imshow("Text Detection", bubble_segmentation)
#    cv2.waitKey(0)

    text_detected_image = image

    return bubble_segmentation, text_detected_image

if __name__ == '__main__':
    # load the pre-trained EAST text detector
    net_EAST = cv2.dnn.readNet('./checkpoints/frozen_east_text_detection.pb')

    # get all data path
    file_list = []
    in_dir_name = './data/comic_img/train/'
    for filename in os.listdir(in_dir_name):
        if filename.lower().endswith('.png'):
            file_list.append(os.path.join(in_dir_name, filename))
    in_dir_name = './data/comic_img/validation/'
    for filename in os.listdir(in_dir_name):
        if filename.lower().endswith('.png'):
            file_list.append(os.path.join(in_dir_name, filename))


    for img_path in file_list:
        Speech_Bubble_segmentation(img_path=img_path)