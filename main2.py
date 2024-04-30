import tensorflow as tf
import cv2
import numpy as np
import json
from datetime import datetime
import pygame
import sys
import subprocess
import platform
#import pywifi
#from pywifi import const
import random 
import string 
import hashlib


def recognizer():
    data_list = []
    model_liver = tf.keras.models.load_model("trained_model.h5")  

    cap = cv2.VideoCapture(0)

    liver_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "trained_model.h5")

    while True:
        ret, frame = cap.read()  

        predictions_gender = model_liver.predict()
        #díky tomuto to funguje 
        female_probability = predictions_gender[0][0] >0.5


        data_list.append({

                    "date/time": datetime.now().isoformat(),                          
                        })

        cv2.putText(frame, f'Gender: {}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)



        cv2.imshow("Live detekce jater ", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            with open("data.json", "a") as json_file:
                json.dump(data_list, json_file, indent=4)
                json_file.write("\n")
            break


    cap.release()
    cv2.destroyAllWindows()
recognizer()