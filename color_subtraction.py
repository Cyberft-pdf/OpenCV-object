import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



"""
-----------------------------------------------------------------------------------------------------------------------

    This code snippet uses OpenCV to perform background subtraction and contour detection on a video feed from a webcam

-----------------------------------------------------------------------------------------------------------------------    

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)


cap = cv2.VideoCapture(0) 

while True:

    ret, frame = cap.read()
    if not ret:
        break
    

    fg_mask = bg_subtractor.apply(frame)
    
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground Mask', fg_mask)
    

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
"""



"""
----------------------------------------------------------------


This code snippet demonstrates how to detect pink objects

---------------------------------------------------------------- 


image = cv2.imread("9.jpg")

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_pink = np.array([120, 80, 80]) 
upper_pink = np.array([200, 255, 255])  

mask = cv2.inRange(hsv_image, lower_pink, upper_pink)

pink_objects = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Original Image", image)
cv2.imshow("Pink Objects", pink_objects)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""




"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------    

This code snippet applies a sliding window approach to detect areas of an image that match a certain color range, specifically targeting dark blue regions

---------------------------------------------------------------------------------------------------------------------------------------------------------------    


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


"""



