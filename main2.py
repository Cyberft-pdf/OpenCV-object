import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



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
            face = cv2.resize(face,(200,200))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)


        cv2.imshow("Live Detekce", frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    cap.release()
    cv2.destroyAllWindows()

recognizer()
"""


"""

import cv2
import numpy as np

# Načtení obrázku
image = cv2.imread("9.jpg")

# Převod obrázku do barevného prostoru HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Určení rozsahu barev pro růžovou barvu
lower_pink = np.array([120, 80, 80])  # Dolní mez pro odstín, sytost a hodnotu
upper_pink = np.array([200, 255, 255])  # Horní mez pro odstín, sytost a hodnotu

# Vytvoření masky pro růžovou barvu
mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

# Aplikace masky na původní obrázek
pink_objects = cv2.bitwise_and(image, image, mask=mask)

# Zobrazení původního obrázku a detekovaných objektů s růžovou barvou
cv2.imshow("Original Image", image)
cv2.imshow("Pink Objects", pink_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""



import cv2

image = cv2.imread("9.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

window_size = (50, 50)
step_size = 10

target_color = (255, 0, 0)

color_tolerance = 50
lower_dark_blue = np.array([90, 50, 50])
upper_dark_blue = np.array([120, 255, 255])

dark_blue_mask = cv2.inRange(image_rgb, lower_dark_blue, upper_dark_blue)

# Aplikace masky na obrázek
dark_blue_image = cv2.bitwise_and(image, image, mask=dark_blue_mask)
cv2.imshow("Sliding Window", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()






