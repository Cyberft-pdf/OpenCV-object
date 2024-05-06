"""
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)


cap = cv2.VideoCapture(0) 

while True:

    ret, frame = cap.read()
    if not ret:
        break
    

    fg_mask = bg_subtractor.apply(frame)
    
    #nalezaní kontur
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    




    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""

import cv2
import numpy as np
import tensorflow as tf


def recognizer():

    cap = cv2.VideoCapture(0)

    liver_cascade = cv2.CascadeClassifier("/Users/Adela/Desktop/data/classifier/cascade.xml")

    while True:
        ret, frame = cap.read()  

        liver = liver_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in liver:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face,(400,200))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)


        cv2.imshow("Live Detekce", frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()

recognizer()