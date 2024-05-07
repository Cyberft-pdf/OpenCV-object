import cv2

# Načtení obrázku
image = cv2.imread("9.jpg")

# Převedení na černobílý obrázek
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplikace práhování
_, thresholded_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

# Zobrazení výsledku
cv2.imshow("Segmented Image", thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()