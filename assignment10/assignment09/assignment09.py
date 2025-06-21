import cv2
import os

#creamos nuestra carpeta para guardar los frames
os.makedirs('frames_guardados_4', exist_ok=True)

#Líneas para elegir el video a leer.

#cap = cv2.VideoCapture('video_sarah.mp4')
#cap = cv2.VideoCapture('prueba_2.mp4') #video de prueba 2
cap = cv2.VideoCapture(0) #cámara web

#vamos a hacer que detecte el rostro usando Viola & James
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
found_face = False #booleano para saber si hay rostro detectado
template = None #template inicializado en Nada

t= 0
#Primero leemos el frame y detectamos el rostro 
while True:
    ret, f_t = cap.read()
    if not ret: #si no hay frame leído salimos del bucle
        break
    t += 1
    
    #Si sí, aplicamos template matching
    if found_face:
        #aplicamos el template matching
        resultado = cv2.matchTemplate(f_t, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)
        
        if max_val > 0.7:
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(f_t, top_left, bottom_right, (0, 255, 0), 2) #rectangulo azul
            #cv2.putText(f_t, "Rostro detectado", (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #wait key
            cv2.waitKey(1)
        else:
            found_face = False
            template = None
    if not found_face:
        #nuevamente detectamos Viola & Jones
        img_grises = cv2.cvtColor(f_t, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_grises, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            
            x, y, w, h = faces[0] #usar faces[0] me asegura que solo un rostro se detecte
            template = f_t[y:y+h, x:x+w]
            found_face = True
            cv2.rectangle(f_t, (x, y), (x + w, y + h), (0, 255, 0), 2) #rectangulo verde
            #cv2.putText(f_t, "Rostro re-detectado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Rostro detectado en el frame t = {t}")
            
        else:
            print(f"No se detectó rostro en el frame t = {t}")
    #guardamos frames
    if t % 50 == 0:
        cv2.imwrite(f'frames_guardados_4/frame_{t}.jpg', f_t)
    #mostramos el frame
    cv2.imshow('Seguimiento', f_t)
    
    #salimos del programa si se presiona q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
            
