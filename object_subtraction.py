
import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt


"""

import cv2
import numpy as np


cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:

    ret, frame = cap.read()  

    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        cv2.imshow("Live Detekce", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


cap.release()
cv2.destroyAllWindows()

"""
"""
import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

cesta = "liver_gif.gif"

cap = cv2.VideoCapture(cesta)
# umožněje odebrat pozadí (vytváří další objek)
backSub = cv2.createBackgroundSubtractorMOG2()


if not cap.isOpened():
    print("Error")
    while cap.isOpened():
          ret, frame = cap.read()
          if ret:
            fg_mask = backSub.apply(frame)
            #najdi contours
            contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            #vybarvi je 
            frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            #vytvoř masku 
            retval, mask_thresh = cv2.threshold( fg_mask, 180, 255, cv2.THRESH_BINARY)


            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            min_contour_area = 500 
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            

            frame_out = frame.copy()
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            
            cv2.imshow('Frame_final', frame_out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

cap.release()
cv2.destroyAllWindows()
"""
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


cap = cv2.VideoCapture(0)


bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60, detectShadows=False)

if not cap.isOpened():
    print("Error opening video file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
    
        fg_mask = bg_subtractor.apply(frame)
       
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  
        spojnice = []

        for i in range(len(contours)):
            if hierarchy[0][i][3] != -1:
                #řešení spojů => je to řešené pomocí hierarchie 
                if hierarchy[0][i][2] == -1:
                    rodic_contour = contours[hierarchy[0][i][3]]
                    spojnice.append((rodic_contour, contours[i]))


        fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        for rodic_contour, child_contour in spojnice:
            cv2.drawContours(output_image, [rodic_contour, child_contour], -1, (0, 255, 0), 3)


        output_image = fg_mask_color.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)) 
        fg_mask_eroded = cv2.erode(fg_mask, kernel, iterations=1)  

        fg_mask_dilated = cv2.dilate(fg_mask, kernel, iterations=1)
       
        frame_out = frame.copy()
        for cnt in contours:
           for point in cnt:
                x, y = point[0]
                cv2.circle(frame_out, (x, y), 1, (0, 0, 255), 1)  
      
        cv2.imshow("se spojem", output_image )
        cv2.imshow("Frame_final", frame_out)
        cv2.imshow("eroded", fg_mask_eroded)
        cv2.imshow("dilated", fg_mask_dilated)
  
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
