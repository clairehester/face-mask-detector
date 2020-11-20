from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from PIL import Image,ImageEnhance
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import imutils
import time
import glob
import random
import cv2
import os

st.set_option('deprecation.showfileUploaderEncoding', False)
def imagepreds(image):

    # load in models
    face_detector = MTCNN(min_face_size=15)
    model = load_model('face_mask_detector.h5')

    
    orig = image.copy()
    (im_h, im_w) = image.shape[:2]
    # blob takes in the image, scale factor, output size, and RGB means for subtraction.
    # ImageNet means are used as the default. 
    detections = face_detector.detect_faces(image)

    for detected_face in detections:
        confidence = detected_face['confidence']
        if confidence >= 0.5:
            x, y, w, h = detected_face['box']
            (x, y) = (max(0, x), max(0, y))
            (x2, y2) = (min(im_w - 1, x+w), min(im_h - 1, y+h))
            face = image[y:y2, x:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = preprocess_input(face)

            mask_weared_incorrect, with_mask, without_mask = model.predict(face)[0]
            # determine the class label and color we'll use to draw the bounding box and text
            if max([mask_weared_incorrect, with_mask, without_mask]) == with_mask:
                label = 'with_mask'
                color = (0, 255, 0)
            elif max([mask_weared_incorrect, with_mask, without_mask]) == without_mask:
                label = 'without_mask'
                color = (0, 0, 255)
            else:
                label = 'mask_worn_incorrectly'
                color = (255, 0, 0)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max([mask_weared_incorrect, with_mask, without_mask]) * 100)
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (x, y), (x2, y2), color, 2)

    return st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Predictions', width=720)
            

def videopreds():

    def detect_and_predict_mask(frame, faceNet, maskNet):
        # grab the dimensions of the frame and then construct a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
        faceNet.setInput(blob)
        detections = faceNet.forward()
        # initialize list of faces, locations, and predictions
        faces = []
        locs = []
        preds = []
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract confidence 
            confidence = detections[0, 0, i, 2]
            # filter out weak detections
            if confidence > 0.5:
                # compute the (x, y)-coordinates of the bounding box 
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)
        # return a tuple of the face locations and their corresponding
        # locations
        return (locs, preds)



    prototxtPath = os.path.sep.join(['./opencv_face_detector/', "deploy.prototxt"])
    weightsPath = os.path.sep.join(['./opencv_face_detector/', "res10_300x300_ssd_iter_140000.caffemodel"])
    face_detector = cv2.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    model = load_model('face_mask_detector.h5')

    def get_cap():
        return cv2.VideoCapture(0)

    cap = get_cap()

    frameST = st.empty()

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        ret, frame = cap.read()
        # run our function to make predictions
        (locs, preds) = detect_and_predict_mask(frame, face_detector, model)



    # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
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
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        # display the output frame
        frameST.image(frame, channels="BGR")
    # key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    # if key == ord("q"):
    #     break

    
    # do a bit of cleanup
    cv2.destroyAllWindows()


def load_image(img):
    im = Image.open(img)
    return im



st.title('Face Mask Detection Classifier')

st.text('Built with Streamlit, Tensorflow, Keras and OpenCV by Claire Hester')

st.header("Select the options from sidebar: ")


def main():

	menu = ['Image Detection', 'Video Detection']
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == 'Image Detection':
		st.subheader('**Face Mask Detection**')
		image_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
        		
		if image_file is not None:
			our_image = Image.open(image_file)
			st.text("Original Image")
			st.image(our_image)

		enhance_type = st.sidebar.radio("Enhance Type", ["Original", "Detected"])
		if enhance_type == 'Detected':
			new_img = np.array(our_image.convert('RGB'))
			img = cv2.cvtColor(new_img,3)
			detected_image = imagepreds(img)
			detected_image

	elif choice == 'Video Detection':
		st.subheader('**Face Mask Detection**')
		videopreds()

if __name__ == '__main__':
    main()