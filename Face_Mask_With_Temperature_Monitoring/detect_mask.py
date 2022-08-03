# Course: Engineering Capstone Project
# Project: Face mask detection with temperature monitoring
# Group No.: 2
# Name and Student Id: Sejal Chovatiya - 8740076
#                      Divyesh Korat - 8716950
#                      Utkarsh Purohit - 8739830
# Low level software working: For this we have created program to check person is wearing mask or not from webcam.
# Reference: https://www.youtube.com/watch?v=QRtBs1SXToc
#            https://drive.google.com/drive/folders/1qjgodqLIOdy6c63BFJkLw6lDTzJ8Gz8N

from smbus2 import SMBus
from mlx90614 import MLX90614

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from os.path import dirname, join
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import datetime

from smbus2 import SMBus
from mlx90614 import MLX90614

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # convert it from BGR to RGB channel and ordering, resize
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"deploy.protext"
weightsPath = r"res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("Starting the CAMERA...")
vs = VideoStream(src=0).start()
# vs1 = cv2.VideoCapture(0)

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=1024)

    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        bus = SMBus(1)
        sensor = MLX90614(bus, address=0x5A)

        amb_temp = sensor.get_amb_temp()
        amb_temp_two = "{:.2f}".format(amb_temp)

        obj_temp = sensor.get_obj_temp()
        obj_temp_two = "{:.2f}".format(obj_temp)

        # print("Ambient Temperature :", amb_temp_two)
        # print("Object Temperature :", obj_temp_two)
        
        bus.close()

        # draw bounding box and text
        label = "Mask Detected" if mask > withoutMask else "No Mask"
        label1 = "{} (Temp : {:.2f})".format(label, obj_temp)
        label2 = ''

        if ((label == "No Mask") and (obj_temp < 30.00)):
            color = (255, 0, 0)
            label2 = 'Please Wear a Mask'
            cv2.putText(frame, 
                    label2, 
                    (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (255, 0, 0), 
                    2)
            now = datetime.datetime.now()
            print("Current date and time: ")
            print(str(now))
            
        if obj_temp > 30.00:
            color = (0, 0, 255)
            label2 = 'Temperature is high, You are not allow to enter'
            cv2.putText(frame, 
                    label2, 
                    (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 0, 255), 
                    2)

        if ((label == "Mask Detected") and (obj_temp < 30.00)):
            color = (0, 255, 0)
            label2 = 'You are allow to enter'
            cv2.putText(frame, 
                    label2, 
                    (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                    (0, 255, 0), 
                    2)
        
        # include the probability in the label
        # label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
       
        print(label1)
        print(label2)        
        
        # display the label and bounding box rectangle on the output
        cv2.putText(frame, label1, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)

    cv2.imshow("Frame", frame)
    
    # creating 'q' as the quit button for the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# do a bit of cleanup

# vs1.release()
cv2.destroyAllWindows()
vs.stop()

