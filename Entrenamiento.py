#------------------------------- Importamos librerías ---------------------------------
import cv2
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Preprocesar imágenes
from tensorflow.keras import optimizers  # Optimizadores
from tensorflow.keras.models import Sequential  # Modelos secuenciales
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D  # Capas
from tensorflow.keras import backend as K  # Backend de Keras

# Limpiamos la sesión de Keras
K.clear_session()

#--------------------------------- Rutas de los datos ---------------------------------
datos_entrenamiento = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/fotos/entrenamiento'
datos_validacion = 'C:/Users/ALUMNO-ETIA/Desktop/senati/vision python/manos/fotos/validacion'

# Parámetros
iteraciones = 20  # Número de iteraciones para ajustar nuestro modelo
altura, longitud = 200, 200  # Tamaño de las imágenes de entrenamiento
batch_size = 1  # Asegúrate de que sea un entero
pasos = int(300 / batch_size)  # Convertir a entero
pasos_validacion = int(300 / batch_size)  # Convertir a entero
filtrosconv1 = 32
filtrosconv2 = 64  # Número de filtros que vamos a aplicar en cada convolución
tam_filtro1 = (3, 3)
tam_filtro2 = (2, 2)  # Tamaños de los filtros
tam_pool = (2, 2)  # Tamaño del filtro en max pooling
clases = 2  # Mano abierta y cerrada
lr = 0.0005  # Tasa de aprendizaje

# Pre-Procesamiento de las imágenes
preprocesamiento_entre = ImageDataGenerator(
    rescale=1. / 255,  # Normalización
    shear_range=0.3,  # Transformaciones
    zoom_range=0.3,
    horizontal_flip=True  # Invertir imágenes
)

preprocesamiento_vali = ImageDataGenerator(
    rescale=1. / 255  # Solo normalización para validación
)

# Cargamos los datos
imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical',
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

# Creamos la red neuronal convolucional (CNN)
cnn = Sequential()
cnn.add(Conv2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))  # Primera capa
cnn.add(MaxPooling2D(pool_size=tam_pool))  # Max pooling
cnn.add(Conv2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))  # Segunda capa
cnn.add(MaxPooling2D(pool_size=tam_pool))

# Aplanamos y agregamos capas densas
cnn.add(Flatten())
cnn.add(Dense(256, activation='relu'))
cnn.add(Dropout(0.5))  # Regularización
cnn.add(Dense(clases, activation='softmax'))  # Capa de salida

# Compilamos el modelo
optimizar = optimizers.Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

# Entrenamos nuestra red
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)


# Guardamos el modelo
cnn.save('Modelo.h5')
cnn.save_weights('pesos.weights.h5')  # Cambia aquí


