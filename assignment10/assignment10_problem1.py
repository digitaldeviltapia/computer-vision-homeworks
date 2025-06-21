import cv2
import os
def iou(caja1, caja2):
    x1, y1, w1, h1 = caja1
    x2, y2, w2, h2 = caja2
    
    #tomaremos las coordenadas de intersección
    x_i1 = max(x1, x2)
    y_i1 = max(y1, y2)
    x_i2 = min(x1 + w1, x2 + w2)
    y_i2 = min(y1 + h1, y2 + h2)
    
    #ahora el área de intersección
    ancho_interseccion = max(0, x_i2 - x_i1)
    alto_interseccion = max(0, y_i2 - y_i1)
    area_interseccion = ancho_interseccion * alto_interseccion
    
    #ahora el área de unión
    area_1 = w1 * h1
    area_2 = w2 * h2
    area_union = area_1 + area_2 - area_interseccion
    
    #IoU
    return area_interseccion / area_union if area_union else 0.0 

def distancia_centros(caja1, caja2, tol_iou = 0.3, distancia_tol = 40):
    if iou(caja1, caja2) > tol_iou:
        return True
    #si no, calculamos la distancia entre centros
    x1, y1, w1, h1 = caja1
    x2, y2, w2, h2 = caja2
    
    #tomamos los centros de las cajas
    cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
    cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
    
    distancia = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    
    return distancia < distancia_tol

## -----Aquí comienza el programa después de todas las
## funcones que pusimos antes


# creamos nuestra carpeta para guardar los frames
os.makedirs('caras_tapia', exist_ok=True)

# elegimos vídeo a leer

cap = cv2.VideoCapture('prueba_2.mp4')
#cap = cv2.VideoCapture('walking_london.mp4') #video de prueba 2
#cap = cv2.VideoCapture(0) #cámara web
#cap = cv2.VideoCapture('video1_disco.mp4') #video de prueba 3
#cap = cv2.VideoCapture('video_dancing.mp4') #video de prueba 4
#cap = cv2.VideoCapture('video2_disco.mp4') #video de prueba 5
#cap = cv2.VideoCapture('video_walking_nuevo_hd.mp4') #video walking london

#fps del vid
fps = cap.get(cv2.CAP_PROP_FPS)


if fps == 0:
    fps = 30
wait_time = int(1000 / (fps))  # dividir entre fps

#vamos a hacer que detecte el rostro usando Viola & James
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

detected_faces = [] #lista para guardar los rostros detectados
total_faces = 0 #contador de rostros detectados

found_face = False
template    = None

t= 0 #contador de frames


    #Primero leemos el frame y detectamos el rostro 
while True:
    ret, f_t = cap.read()
    if not ret: #si no hay frame leído salimos del bucle
        break
    t += 1
    # template matching
    if found_face and template is not None:
        #aplicamos el template matching
        resultado = cv2.matchTemplate(f_t, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)
        
        if max_val > 0.7:
            top_left = max_loc
            bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
            cv2.rectangle(f_t, top_left, bottom_right, (0, 255, 0), 2) #rectangulo azul
            
        else:
            found_face = False
            template = None
    else:
        #nuevamente detectamos Viola & Jones
        img_grises = cv2.cvtColor(f_t, cv2.COLOR_BGR2GRAY) #convertimos a gris
        
        faces = face_cascade.detectMultiScale(img_grises, scaleFactor=1.1, minNeighbors=8, minSize=(100, 100))
        
        if len(faces) > 0:
            #aquí modificaremos para que se detecten todos los rostros en escena
        
            for (x, y, w, h) in faces:
                #primero region de interés
                roi_gris = img_grises[y:y+h, x:x+w]
                
                #buscamos ojos
                eyes = eye_cascade.detectMultiScale(roi_gris, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                if len(eyes) == 0:
                    continue #es un falso positivo
            
                #si sí hay ojos, procedemos como antes
                cv2.rectangle(f_t, (x, y), (x+w, y+h), (0, 255, 0), 2) 
                dist_tol_nueva = max(w, h) * 0.6
                rostro_nuevo = True
                for existente in detected_faces:
                    if distancia_centros((x, y, w, h), existente, tol_iou = 0.01, distancia_tol = dist_tol_nueva):
                        rostro_nuevo = False
                        break
                if rostro_nuevo:
                    detected_faces.append((x, y, w, h))
                    total_faces += 1
                    template = f_t[y:y+h, x:x+w]
                    found_face = True
                    print(f"Rostro detectado en el frame t = {t}")
                    #imprimimos el total de rostros detectados
                    print(f"Van {total_faces} rostros detectados")
                    #guardamos frames
                    cv2.imwrite(f'caras_tapia/rostro_{total_faces}.jpg', f_t)
        else:
            #print(f"No se detectó rostro en el frame t = {t}")
            pass
    
    #mostramos el frame
    
    cv2.putText(f_t, f'Rostros detectados: {total_faces}', (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Seguimiento', f_t)
    
    #salimos del programa si se presiona q
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break
print(f"Total de rostros distintos detectados: {total_faces}")
cap.release()
cv2.destroyAllWindows()
