import cv2
import mediapipe as mp
import os

#----------------------------- Creamos la carpeta donde almacenaremos el entrenamiento ---------------------------------
nombre = 'Mano_Izquierda'
direccion = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/fotos/validacion'
carpeta = os.path.join(direccion, nombre)
if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

# Asignamos un contador para el nombre de las fotos
cont = 0

# Leemos la cámara
cap = cv2.VideoCapture(0)

#---------------------------- Creamos un objeto que va a almacenar la detección y el seguimiento de las manos ------------
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()  # Primer parámetro, FALSE para que no haga la detección 24/7

#---------------------------------- Método para dibujar las manos ---------------------------
dibujo = mp.solutions.drawing_utils  # Con este método dibujamos 21 puntos críticos de la mano

while True:
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista vamos a almacenar las coordenadas de los puntos

    if resultado.multi_hand_landmarks:  # Si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks:  # Buscamos la mano dentro de la lista de manos
            for id, lm in enumerate(mano.landmark):  # Obtenemos la información de cada mano
                alto, ancho, _ = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if len(posiciones) != 0:
                pto_i5 = posiciones[9]  # Punto central (dedo pulgar)
                x1, y1 = max(0, pto_i5[1] - 80), max(0, pto_i5[2] - 80)  # Asegurarse de que no sea negativo
                x2, y2 = min(ancho, x1 + 160), min(alto, y1 + 160)  # Asegurar que no sobrepase los límites

                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Guardar la imagen de la mano
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)  # Redimensionar
                cv2.imwrite(os.path.join(carpeta, f"Mano_{cont}.jpg"), dedos_reg)  # Guardar imagen
                cont += 1  # Incrementar el contador

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= 300:  # Salir si se presiona ESC o se toman 300 fotos
        break

cap.release()
cv2.destroyAllWindows()
