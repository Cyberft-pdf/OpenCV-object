import cv2
import numpy as np
import tensorflow as tf

# Načtení natrénovaného modelu
model = tf.keras.models.load_model("trained_model.h5")

# Inicializace kamery
cap = cv2.VideoCapture(0)  # 0 pro výběr první kamery, můžete změnit podle potřeby

while True:
    # Načtení snímku z kamery
    ret, frame = cap.read()
    if not ret:
        break

    # Změna velikosti snímku na 200x200 pixelů, kterou vyžaduje váš model
    frame_resized = cv2.resize(frame, (200, 200))

    # Předzpracování snímku - normalizace pixelů do rozsahu 0-1
    frame_resized = frame_resized / 255.0

    # Rozšíření dimenze snímku pro predikci pomocí modelu
    frame_expanded = np.expand_dims(frame_resized, axis=0)

    # Provádění predikce na snímku pomocí modelu
    prediction = model.predict(frame_expanded)

    # Získání výsledné třídy (0 nebo 1) a pravděpodobnosti
    predicted_class = int(np.round(prediction)[0][0])
    probability = prediction[0][0]

   # Zobrazení výsledků na snímku
    if predicted_class == 1:
        cv2.putText(frame, "Jatra nalezena (pravděpodobnost: {:.2f})".format(probability), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Získání souřadnic detekovaných jater a jejich velikosti
        x, y, w, h = 50, 50, 100, 100  # Nahraďte těmito hodnotami souřadnic a velikostí, které odpovídají vašim detekcím jater
        
        # Vykreslení čtverce kolem detekovaných jater
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    else:
        cv2.putText(frame, "Jatra nenalezena (pravděpodobnost: {:.2f})".format(probability), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    # Zobrazení snímku
    cv2.imshow("Liver Detection", frame)

    # Ukončení smyčky, pokud uživatel stiskne klávesu 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Uvolnění kamery a zavření všech otevřených oken
cap.release()
cv2.destroyAllWindows()

