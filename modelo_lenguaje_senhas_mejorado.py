import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import os
import cv2
import time

# Configurar semilla para reproducibilidad
np.random.seed(42)
tf.random.set_seed(42)

# Crear directorio para logs y checkpoints
timestamp = int(time.time())
log_dir = f"logs/fit/{timestamp}"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Cargar datos
print("Cargando datos de entrenamiento...")
train_data = pd.read_csv('sign_mnist_train.csv')
print("Cargando datos de prueba...")
test_data = pd.read_csv('sign_mnist_test.csv')

# Separar etiquetas y características
y_train = train_data['label']
X_train = train_data.drop('label', axis=1)
y_test = test_data['label']
X_test = test_data.drop('label', axis=1)

# Convertir las etiquetas a one-hot encoding
y_train = to_categorical(y_train, num_classes=25)  # 25 letras (sin J ni Z que requieren movimiento)
y_test = to_categorical(y_test, num_classes=25)

# Preprocesar las imágenes
# Normalizar y reshapear las imágenes a 28x28 píxeles
X_train = X_train.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Separar un conjunto de validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Crear el modelo con capas de preprocesamiento para aumentación de datos
def crear_modelo_mejorado():
    # Regularización L2 para capas convolucionales y densas
    conv_regularizer = l2(0.0001)
    dense_regularizer = l2(0.0005)
    
    model = Sequential([
        # Capas de aumento de datos integradas en el modelo
        RandomFlip("horizontal", input_shape=(28, 28, 1)),
        RandomRotation(0.1),
        RandomZoom(0.1),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
        
        # Primera capa convolucional
        Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(32, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),  # Aumentado de 0.2 a 0.3
        
        # Segunda capa convolucional
        Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(64, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),  # Aumentado de 0.3 a 0.4
        
        # Tercera capa convolucional
        Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Conv2D(128, kernel_size=(3, 3), padding='same', kernel_regularizer=conv_regularizer),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),  # Aumentado de 0.4 a 0.5
        
        # Aplanar y capas densas
        Flatten(),
        # Reducir parámetros en capas densas
        Dense(192, kernel_regularizer=dense_regularizer),  # Reducido de 256 a 192
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.6),  # Aumentado de 0.5 a 0.6
        Dense(25, activation='softmax')  # 25 clases para las letras del lenguaje de señas
    ])
    
    # Compilar el modelo con una tasa de aprendizaje fija para evitar errores
    # Usaremos ReduceLROnPlateau para ajustar la tasa durante el entrenamiento
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Crear el modelo mejorado
modelo = crear_modelo_mejorado()
modelo.summary()

# Callbacks mejorados para el entrenamiento
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,            # Aumentado para dar más oportunidad al modelo
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,              # Reduce la tasa por un factor de 0.2
    patience=5,              # Esperar 5 épocas sin mejora
    min_lr=1e-7,             # Tasa mínima más baja
    verbose=1
)

# Checkpoint para guardar el mejor modelo
checkpoint = ModelCheckpoint(
    os.path.join(checkpoint_dir, 'modelo_senhas_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.h5'),
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# Guardar también el modelo final con un nombre fijo para facilitar su uso
final_model_checkpoint = ModelCheckpoint(
    'modelo_lenguaje_senhas_mejorado.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# TensorBoard para visualización del entrenamiento
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

# Entrenar el modelo con configuración mejorada
epochs = 40  # Aumentar epochs para ver mejor el comportamiento
batch_size = 64

print("Entrenando el modelo mejorado...")
history = modelo.fit(
    X_train, y_train,  # Ya no usamos el generador, ahora tenemos aumentación integrada
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr, checkpoint, final_model_checkpoint, tensorboard_callback],
    verbose=1
)

# Evaluar el modelo en los datos de prueba
print("Evaluando el modelo...")
test_loss, test_acc = modelo.evaluate(X_test, y_test)
print(f"Precisión en el conjunto de prueba: {test_acc:.4f}")

# Visualizar el rendimiento del modelo
plt.figure(figsize=(14, 5))

# Gráfico de precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo mejorado')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Gráfico de pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo mejorado')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('rendimiento_modelo_mejorado.png', dpi=300)
plt.show()

# Guardar el modelo final
modelo.save('modelo_lenguaje_senhas_mejorado.h5')
print("Modelo mejorado guardado como 'modelo_lenguaje_senhas_mejorado.h5'")

# Analizar la matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Obtener predicciones
y_pred = modelo.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Generar matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Visualizar matriz de confusión
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list('ABCDEFGHIKLMNOPQRSTUVWXY'),
            yticklabels=list('ABCDEFGHIKLMNOPQRSTUVWXY'))
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.savefig('matriz_confusion.png', dpi=300)
plt.show()

# Imprimir informe de clasificación
report = classification_report(y_true, y_pred_classes, 
                              target_names=list('ABCDEFGHIKLMNOPQRSTUVWXY'))
print("Informe de clasificación:")
print(report)

# Función para predecir una imagen individual (mantenemos esta funcionalidad)
def predecir_letra(ruta_imagen=None, imagen_array=None):
    # Mapeo de índices a letras (sin J y Z)
    letras = 'ABCDEFGHIKLMNOPQRSTUVWXY'
    
    if ruta_imagen is not None:
        # Cargar y preprocesar la imagen
        img = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
    elif imagen_array is not None:
        img = imagen_array.reshape(28, 28)
    else:
        return "Error: Proporciona una ruta de imagen o un array de imagen"
    
    # Normalizar y reshapear para el modelo
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    
    # Predecir
    pred = modelo.predict(img)
    indice_letra = np.argmax(pred)
    
    # Convertir a letra
    letra = letras[indice_letra]
    confianza = np.max(pred) * 100
    
    return letra, confianza

print("¡El modelo mejorado está listo para identificar letras del lenguaje de señas americano!") 