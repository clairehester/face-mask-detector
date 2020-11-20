#!/usr/bin/env python
# coding: utf-8


# import packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os



# Create a function for easy deployment based on 03b_face_detector_mtcnn.ipynb
# Params are the frame captured through video, face detector model, and mask detector model
def detect_and_predict_mask(frame, face_detector, model):
    # grab frame dimensions and convert color
    (im_h, im_w) = frame.shape[:2]
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_faces = face_detector.detect_faces(new_frame)
    # initialize list of faces, locations, and predictions
    faces = []
    locs = []
    preds = []
    
    # loop over the detections
    for detected_face in detected_faces:
        # extract confidence
        confidence = detected_face['confidence']
        #filter out weak detections
        if confidence >= 0.5:
            # compute (x,y)-coordinates of the bounding box
            x, y, w, h = detected_face['box']
            # ensure bounding box falls within frame dimensions
            (x, y) = (max(0, x), max(0, y))
            (x2, y2) = (min(im_w - 1, x+w), min(im_h - 1, y+h))

            #extract the face ROI and preprocess
            face = frame[y:y2, x:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((x, y, x2, y2))
    
    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        
        faces = np.array(faces, dtype="float32")
        preds = model.predict(faces, batch_size=32)
    # return a tuple of the face locations and their corresponding predictions
    return (locs, preds)


print("[INFO] loading face detector model...")
face_detector = MTCNN(min_face_size=15)
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model('face_mask_detector.h5')
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# log start time and start frame count
start_time = time.time()
frame_count = 0

# loop over the frames from the video stream
while True:
    # track frame count
    frame_count +=1
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # run function to make predictions
    (locs, preds) = detect_and_predict_mask(frame, face_detector, model)



# loop over the detected face locations and their corresponding locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (x, y, x2, y2) = box
        mask_weared_incorrect, with_mask, without_mask = pred
        # determine the class label and color we'll use to draw
        # the bounding box and text
        if max([mask_weared_incorrect, with_mask, without_mask]) == with_mask:
            label = 'with_mask'
            color = (0, 255, 0)
        elif max([mask_weared_incorrect, with_mask, without_mask]) == without_mask:
            label = 'without_mask'
            color = (0, 0, 255)
        else:
            label = 'mask_worn_incorrectly'
            color = (255, 140, 0)
        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max([mask_weared_incorrect, with_mask, without_mask]) * 100)
        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)


    # display the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Calculate time lapse
time_lapsed = time.time() - start_time
fps = frame_count / time_lapsed
print(f'Total time: {time_lapsed} Frames per second: {fps}')
    
# Clean up
cv2.destroyAllWindows()
vs.stop()





