# Reconocimiento de Lenguaje de Señas Americano con IA

Un sistema de reconocimiento de lenguaje de señas americano con interfaz gráfica que utiliza detección de manos en tiempo real y redes neuronales convolucionales para identificar las letras.

## 📋 Descripción

Este proyecto implementa un sistema de reconocimiento de lenguaje de señas americano con:

- Modelo de deep learning (CNN) que alcanza más del 98% de precisión
- Interfaz gráfica con detección de manos en tiempo real usando MediaPipe
- Procesamiento inteligente de la región de la mano para mejor precisión
- Visualización en tiempo real de las predicciones

El sistema está diseñado para reconocer las 25 letras estáticas del alfabeto de señas americano (excluye J y Z que requieren movimiento).

## 🚀 Instalación

### Requisitos previos

- Python 3.7 o superior
- Webcam

### Dependencias

Instala todas las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

Principales dependencias:
- TensorFlow
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Tkinter (incluido con Python generalmente)

## 💻 Uso

### Ejecutar la interfaz gráfica

La manera más fácil de usar el sistema es mediante el script de ejecución que automáticamente encuentra el mejor modelo disponible:

```bash
python ejecutar_interfaz.py
```

### Entrenar un nuevo modelo

Si deseas entrenar el modelo desde cero con las mejoras implementadas:

```bash
python modelo_lenguaje_senhas_mejorado.py
```

Esto entrenará una red neuronal convolucional utilizando los datos de entrenamiento y guardará el modelo optimizado.

### Uso manual de la interfaz

También puedes ejecutar directamente la interfaz:

```bash
python deteccion_senhas_gui.py
```

Y luego seguir estos pasos:
1. Haz clic en "Cargar Modelo Mejorado" (o selecciona uno manualmente)
2. Haz clic en "Iniciar Captura"
3. Muestra tu mano frente a la cámara
4. La letra predicha aparecerá en la parte derecha de la interfaz

## 🧠 Arquitectura del Modelo

El modelo implementa una red neuronal convolucional (CNN) con:

- Data augmentation integrado (rotación, zoom, traslación)
- 3 bloques convolucionales con BatchNormalization
- Regularización L2 y Dropout creciente para evitar sobreajuste
- Capas densas reducidas para mejorar generalización

### Mejoras específicas:

- Dropout progresivo (0.3 → 0.6)
- Regularización L2 para pesos
- Tasa de aprendizaje adaptativa
- Checkpoints que guardan los mejores modelos

## 📊 Conjunto de datos

El modelo está entrenado con el conjunto de datos Sign MNIST, que incluye imágenes de 28x28 píxeles en escala de grises de letras del lenguaje de señas americano.

## 🔧 Estructura del proyecto

```
├── modelo_lenguaje_senhas_mejorado.py   # Script para entrenar el modelo mejorado
├── deteccion_senhas_gui.py              # Interfaz gráfica con detección de manos
├── ejecutar_interfaz.py                 # Script para ejecutar la interfaz fácilmente
├── prediccion_senhas.py                 # Script para realizar predicciones
├── requirements.txt                     # Dependencias del proyecto
├── .gitignore                           # Archivos a ignorar en el control de versiones
├── rendimiento_modelo.png               # Gráfico de rendimiento del modelo
├── modelo_lenguaje_senhas.h5            # Modelo original guardado
├── modelo_lenguaje_senhas_mejorado.h5   # Modelo mejorado guardado
├── sign_mnist_train.csv                 # Datos de entrenamiento
├── sign_mnist_test.csv                  # Datos de prueba
├── checkpoints/                         # Directorio donde se guardan los modelos durante entrenamiento
└── logs/                                # Registros de entrenamiento para TensorBoard
```

## 🔍 Características de la interfaz

- **Detección de manos en tiempo real**: Utiliza MediaPipe para detectar y rastrear manos
- **Aislamiento automático**: Extrae solo la región de la mano para análisis
- **Visualización de confianza**: Muestra el nivel de confianza de cada predicción
- **Historial de predicciones**: Mantiene registro de las letras detectadas
- **Ajuste de umbral**: Permite configurar el umbral de confianza mínimo
- **Visualización de landmarks**: Opción para mostrar puntos de referencia de la mano
- **Guardado de capturas**: Permite guardar capturas de la cámara

## 🤝 Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Haz un fork del proyecto
2. Crea una rama para tu funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Haz commit de tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Haz push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles.

## 🙏 Agradecimientos

- MediaPipe por la librería de detección de manos
- Sign MNIST por el conjunto de datos de entrenamiento
- La comunidad de lenguaje de señas por inspirar este trabajo

## 📬 Contacto

Si tienes preguntas o comentarios, por favor abre un issue en este repositorio. 