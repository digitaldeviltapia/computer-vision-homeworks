import cv2
import pybgs
import numpy as np
ruta = 'SampleVideo_LowQuality.mp4'
ruta2 = 'video.avi'
cap = cv2.VideoCapture(ruta)
print("Inicializando cámara...")

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Cámara abierta correctamente.")

#llamamos al PBAS (Pixel Based Adaptive Segmenter)
bgs = pybgs.PixelBasedAdaptiveSegmenter()

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.namedWindow("Foreground (PBAS)", cv2.WINDOW_NORMAL)

frame_count = 0

while True:
    ret, frame = cap.read() #ret booleano, frame es el frame que leemos
    if not ret:
        print("No se pudo leer el frame.")
        break

    # pasamos a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar PBAS
    fgmask = bgs.apply(gray)

    # Mostramos dos pantallas, con el vídeo originaly el resultado de la segmentación al mimso tiempo
    cv2.imshow("Frame", frame)
    cv2.imshow("Foreground (PBAS)", fgmask)

    frame_count += 1

    
    key = cv2.waitKey(45)  
    if key == ord('q') or key == 27: #para salir presionamos q o'Esc'
        break

cap.release()
cv2.destroyAllWindows()
print("Fin del programa.")
