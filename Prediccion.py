import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import img_to_array
from keras.models import load_model

# Cargar modelo y pesos
modelo = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/Modelo.h5'
peso = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/pesos.weights.h5'
cnn = load_model(modelo)  # Cargar el modelo
cnn.load_weights(peso)  # Cargar los pesos

# Directorio de imágenes de validación
direccion = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/fotos/validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

# Leer la cámara
cap = cv2.VideoCapture(0)

# Inicializar MediaPipe Pose y Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()

dibujo_pose = mp.solutions.drawing_utils
dibujo_hands = mp.solutions.drawing_utils

# Definir el tamaño de la ventana del video
ancho_video = 750  # Ancho deseado
alto_video = 500   # Alto deseado

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Salir si no se puede leer el frame

    # Redimensionar el marco
    frame = cv2.resize(frame, (ancho_video, alto_video))

    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()

    # Procesar la detección del cuerpo
    resultado_pose = pose.process(color)

    # Procesar la detección de manos
    resultado_hands = hands.process(color)

    # Dibujar los puntos de referencia del cuerpo
    if resultado_pose.pose_landmarks:
        dibujo_pose.draw_landmarks(frame, resultado_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Dibujar las manos y realizar predicciones
    if resultado_hands.multi_hand_landmarks:
        for mano in resultado_hands.multi_hand_landmarks:
            dibujo_hands.draw_landmarks(frame, mano, mp_hands.HAND_CONNECTIONS)

            # Obtener coordenadas de varios puntos de la mano
            posiciones = []
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])

            # Suponiendo que quieres extraer una región alrededor de la palma
            if len(posiciones) > 0:  # Asegurarse de que hay puntos disponibles
                # Usar puntos de la palma (puntos 0 a 4)
                pto_palma_x = [posiciones[0][1], posiciones[1][1], posiciones[2][1], posiciones[3][1], posiciones[4][1]]
                pto_palma_y = [posiciones[0][2], posiciones[1][2], posiciones[2][2], posiciones[3][2], posiciones[4][2]]

                # Calcular los límites de la ROI de la palma
                x1 = min(pto_palma_x) - 50  # Margen de seguridad
                y1 = min(pto_palma_y) - 50
                x2 = max(pto_palma_x) + 50
                y2 = max(pto_palma_y) + 50

                # Verificar si las coordenadas están dentro de los límites
                if x1 >= 0 and y1 >= 0 and x2 <= copia.shape[1] and y2 <= copia.shape[0]:
                    dedos_reg = copia[y1:y2, x1:x2]
                    dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                    x = img_to_array(dedos_reg)
                    x = np.expand_dims(x, axis=0)
                    vector = cnn.predict(x)
                    resultado = vector[0]
                    respuesta = np.argmax(resultado)

                    if respuesta == 1:
                        print(vector, resultado)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                    elif respuesta == 0:
                        print(vector, resultado)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                else:
                    print("Coordenadas fuera de los límites:", x1, y1, x2, y2)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
