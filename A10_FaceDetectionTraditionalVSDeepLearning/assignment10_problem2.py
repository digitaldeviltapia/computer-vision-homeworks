import cv2
import os
import torch
from facenet_pytorch import MTCNN
from deep_sort_realtime.deepsort_tracker import DeepSort

# elegimos dispositivo, por si no hay GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
tracker = DeepSort(max_age=30)
detections = []
# elegir el vídeo a leer

#cap = cv2.VideoCapture('video_sarah.mp4')
cap = cv2.VideoCapture('prueba_2.mp4') #video de prueba 2
#cap = cv2.VideoCapture(0) #cámara web
#cap = cv2.VideoCapture('video1_disco.mp4') #video de prueba 3
#cap = cv2.VideoCapture('video_dancing.mp4') #video de prueba 4
#cap = cv2.VideoCapture('video2_disco.mp4') #video de prueba 5
#cap = cv2.VideoCapture('walking_london.mp4') #video walking london

# creamos nuestra carpeta para guardar los frames
os.makedirs('DL_Tapia1', exist_ok=True)
t = 0
# ahora sí el programa
while True:
    ret, frame = cap.read()
    if not ret:
        break
    t += 1
    #frame to rgb
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #detectamos con MTCNN
    boxes, probs = mtcnn.detect(frame_rgb)
    
   
    #  la lista de detecciones del deeoSORT
    detections = []
    if boxes is not None:
        for (x1, y1, x2, y2), conf in zip(boxes, probs):
            w = x2 - x1
            h = y2 - y1
            
            detections.append(([x1, y1, w, h], float(conf)))
            
    #actualizamos deep sort en tracks
    tracks = tracker.update_tracks(detections, frame=frame)

    #ahora dibujamos las cajas
    for tr in tracks:
        if not tr.is_confirmed():
            continue
        x, y, w, h = tr.to_ltwh()  #coordenadas de la caja
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)  # abajo-derecha
        trackid = tr.track_id
        unique_ids.add(trackid)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, str(trackid), (x1, y1-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    cv2.imwrite(f'DL_Tapia1/rostro_{len(unique_ids)}.jpg', frame)
    
    cv2.putText(frame, f"Total de Rostros Distintos: {len(unique_ids)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("App Deep SORT", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(f"Total de rostros distintos: {len(unique_ids)}")
cap.release()
cv2.destroyAllWindows()
    