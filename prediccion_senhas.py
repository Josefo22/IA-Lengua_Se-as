import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Cargar el modelo entrenado
modelo = load_model('modelo_lenguaje_senhas.h5')

# Mapeo de índices a letras (sin J y Z que requieren movimiento)
letras = 'ABCDEFGHIKLMNOPQRSTUVWXY'

def preprocesar_imagen(ruta_imagen):
    """Preprocesa una imagen para la predicción."""
    img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen desde {ruta_imagen}")
    
    # Redimensionar a 28x28 píxeles
    img = cv2.resize(img, (28, 28))
    
    # Normalizar y reshapear para el modelo
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    return img

def predecir_letra(ruta_imagen):
    """Predice la letra del lenguaje de señas a partir de una imagen."""
    try:
        # Preprocesar la imagen
        img = preprocesar_imagen(ruta_imagen)
        
        # Predecir
        pred = modelo.predict(img, verbose=0)
        indice_letra = np.argmax(pred)
        
        # Convertir a letra
        letra = letras[indice_letra]
        confianza = np.max(pred) * 100
        
        return letra, confianza
    except Exception as e:
        return str(e), 0

def mostrar_prediccion(ruta_imagen):
    """Muestra la imagen y la predicción."""
    # Predecir la letra
    letra, confianza = predecir_letra(ruta_imagen)
    
    # Mostrar la imagen
    img = cv2.imread(ruta_imagen)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.title(f"Predicción: {letra}, Confianza: {confianza:.2f}%")
    plt.axis('off')
    plt.show()
    
    return letra, confianza

def webcam_prediccion():
    """Detecta lenguaje de señas desde la webcam en tiempo real."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la webcam.")
        return
    
    print("Presiona 'q' para salir, 'c' para capturar y predecir.")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: No se pudo leer el frame.")
            break
        
        # Mostrar el frame
        cv2.imshow('Lenguaje de Señas - Webcam', frame)
        
        # Esperar por tecla
        key = cv2.waitKey(1)
        
        # Salir si se presiona 'q'
        if key == ord('q'):
            break
        
        # Capturar y predecir si se presiona 'c'
        if key == ord('c'):
            # Convertir a escala de grises y preprocesar
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (28, 28))
            img = gray.reshape(1, 28, 28, 1).astype('float32') / 255.0
            
            # Predecir
            pred = modelo.predict(img, verbose=0)
            indice_letra = np.argmax(pred)
            letra = letras[indice_letra]
            confianza = np.max(pred) * 100
            
            print(f"Predicción: {letra}, Confianza: {confianza:.2f}%")
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Programa de Predicción de Lenguaje de Señas")
    print("===========================================")
    print("1. Predecir desde imagen de archivo")
    print("2. Utilizar webcam para predicción en tiempo real")
    
    opcion = input("Seleccione una opción (1/2): ")
    
    if opcion == "1":
        ruta_imagen = input("Introduzca la ruta de la imagen: ")
        if os.path.exists(ruta_imagen):
            letra, confianza = mostrar_prediccion(ruta_imagen)
            print(f"Predicción: {letra}, Confianza: {confianza:.2f}%")
        else:
            print(f"Error: No se encontró la imagen en {ruta_imagen}")
    
    elif opcion == "2":
        webcam_prediccion()
    
    else:
        print("Opción no válida.") 