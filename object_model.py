import cv2
import numpy as np
import tensorflow as tf

"""
__________________________________________________________________________________________________

This code snippet performs real-time object detection using a pre-trained TensorFlow model.

__________________________________________________________________________________________________
"""
 



model = tf.keras.models.load_model("trained_model.h5")

cap = cv2.VideoCapture(0)  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (200, 200))

    frame_resized = frame_resized / 255.0

    frame_expanded = np.expand_dims(frame_resized, axis=0)

    prediction = model.predict(frame_expanded)

    predicted_class = int(np.round(prediction)[0][0])
    probability = prediction[0][0]

    if predicted_class == 1:
        cv2.putText(frame, "Jatra nalezena (pravděpodobnost: {:.2f})".format(probability), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        x, y, w, h = 50, 50, 100, 100  
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    else:
        cv2.putText(frame, "Jatra nenalezena (pravděpodobnost: {:.2f})".format(probability), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Liver Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

