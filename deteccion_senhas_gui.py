import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
import threading
import seaborn as sns

# Configurar MediaPipe para la detección de manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Inicializar detector de manos con parámetros mejorados
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,  # 0, 1 o 2, donde 2 es el más preciso pero más lento
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Mapeo de índices a letras (sin J ni Z que requieren movimiento)
letras = 'ABCDEFGHIKLMNOPQRSTUVWXY'

# Variables globales
modelo = None
prediccion_actual = None
confianza_actual = 0
imagen_procesada = None
captura_activa = False
thread_captura = None
historial_predicciones = []

class AplicacionLenguajeSenhas:
    def __init__(self, ventana):
        self.ventana = ventana
        self.ventana.title("Reconocimiento de Lenguaje de Señas - Modelo Mejorado")
        self.ventana.geometry("1200x800")
        self.ventana.configure(bg="#f0f0f0")
        
        # Configurar el estilo
        self.configurar_estilo()
        
        # Crear frames
        self.frame_video = ttk.Frame(ventana, style="Card.TFrame")
        self.frame_video.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.frame_controles = ttk.Frame(ventana, style="Card.TFrame")
        self.frame_controles.pack(side=tk.RIGHT, fill=tk.BOTH, padx=10, pady=10, expand=False, ipadx=10, ipady=10)
        
        # Panel de video
        self.panel_video = ttk.Label(self.frame_video)
        self.panel_video.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Zona de historial de predicciones
        self.frame_historial = ttk.Frame(self.frame_video, style="Card.TFrame")
        self.frame_historial.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        ttk.Label(self.frame_historial, text="Historial de predicciones:", style="Title.TLabel").pack(anchor=tk.W, padx=5, pady=5)
        
        self.historial_text = tk.Text(self.frame_historial, height=5, width=40, font=("Helvetica", 12), state=tk.DISABLED)
        self.historial_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Controles
        ttk.Label(self.frame_controles, text="Controles", style="Title.TLabel").pack(pady=10)
        
        # Botón para cargar modelo
        self.btn_cargar_modelo = ttk.Button(self.frame_controles, text="Cargar Modelo", command=self.cargar_modelo)
        self.btn_cargar_modelo.pack(fill=tk.X, pady=5, padx=10)
        
        # Botón para cargar el modelo mejorado automáticamente
        self.btn_cargar_modelo_mejorado = ttk.Button(
            self.frame_controles, 
            text="Cargar Modelo Mejorado", 
            command=self.cargar_modelo_mejorado
        )
        self.btn_cargar_modelo_mejorado.pack(fill=tk.X, pady=5, padx=10)
        
        # Botón para iniciar/detener captura
        self.btn_iniciar_captura = ttk.Button(self.frame_controles, text="Iniciar Captura", command=self.alternar_captura)
        self.btn_iniciar_captura.pack(fill=tk.X, pady=5, padx=10)
        self.btn_iniciar_captura.config(state=tk.DISABLED)  # Deshabilitado hasta que se cargue un modelo
        
        # Botón para limpiar historial
        self.btn_limpiar_historial = ttk.Button(self.frame_controles, text="Limpiar Historial", command=self.limpiar_historial)
        self.btn_limpiar_historial.pack(fill=tk.X, pady=5, padx=10)
        
        # Botón de guardado de captura
        self.btn_guardar_captura = ttk.Button(self.frame_controles, text="Guardar Captura", command=self.guardar_captura)
        self.btn_guardar_captura.pack(fill=tk.X, pady=5, padx=10)
        self.btn_guardar_captura.config(state=tk.DISABLED)  # Deshabilitado hasta que haya una captura
        
        # Panel de predicción actual
        self.frame_prediccion = ttk.Frame(self.frame_controles, style="Card.TFrame")
        self.frame_prediccion.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(self.frame_prediccion, text="Predicción Actual:", style="Title.TLabel").pack(anchor=tk.W, padx=5, pady=5)
        
        self.letra_predicha = ttk.Label(self.frame_prediccion, text="--", font=("Helvetica", 72, "bold"))
        self.letra_predicha.pack(pady=10)
        
        self.confianza_label = ttk.Label(self.frame_prediccion, text="Confianza: 0.00%")
        self.confianza_label.pack(pady=5)
        
        # Configuraciones avanzadas
        self.frame_config = ttk.LabelFrame(self.frame_controles, text="Configuración", style="Card.TFrame")
        self.frame_config.pack(fill=tk.X, pady=10, padx=10)
        
        # Checkbox para mostrar landmarks
        self.mostrar_landmarks = tk.BooleanVar(value=True)
        self.check_landmarks = ttk.Checkbutton(
            self.frame_config, 
            text="Mostrar landmarks",
            variable=self.mostrar_landmarks
        )
        self.check_landmarks.pack(anchor=tk.W, padx=10, pady=5)
        
        # Checkbox para procesar solo cuando hay mano
        self.procesar_solo_mano = tk.BooleanVar(value=True)
        self.check_solo_mano = ttk.Checkbutton(
            self.frame_config, 
            text="Procesar solo con mano detectada",
            variable=self.procesar_solo_mano
        )
        self.check_solo_mano.pack(anchor=tk.W, padx=10, pady=5)
        
        # Slider para el umbral de confianza
        ttk.Label(self.frame_config, text="Umbral de confianza:").pack(anchor=tk.W, padx=10, pady=(5, 0))
        self.umbral_confianza = tk.DoubleVar(value=70.0)
        self.slider_confianza = ttk.Scale(
            self.frame_config,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.umbral_confianza
        )
        self.slider_confianza.pack(fill=tk.X, padx=10, pady=(0, 5))
        self.label_umbral = ttk.Label(self.frame_config, text="70.0%")
        self.label_umbral.pack(anchor=tk.E, padx=10)
        
        # Actualizar el texto del umbral cuando cambia el slider
        def actualizar_umbral(event):
            self.label_umbral.config(text=f"{self.umbral_confianza.get():.1f}%")
        
        self.slider_confianza.bind("<Motion>", actualizar_umbral)
        
        # Estado del modelo
        self.frame_estado = ttk.Frame(self.frame_controles)
        self.frame_estado.pack(fill=tk.X, pady=10, padx=10)
        
        self.estado_modelo_label = ttk.Label(self.frame_estado, text="Estado del modelo: No cargado", foreground="red")
        self.estado_modelo_label.pack(anchor=tk.W)
        
        # Indicaciones
        self.frame_ayuda = ttk.Frame(self.frame_controles, style="Card.TFrame")
        self.frame_ayuda.pack(fill=tk.X, pady=10, padx=10)
        
        ttk.Label(self.frame_ayuda, text="Cómo usar:", style="Title.TLabel").pack(anchor=tk.W, padx=5, pady=5)
        
        instrucciones = [
            "1. Carga el modelo mejorado",
            "2. Inicia la captura de video",
            "3. Muestra tu mano en la cámara",
            "4. La predicción de la letra aparecerá a la derecha"
        ]
        
        for instruccion in instrucciones:
            ttk.Label(self.frame_ayuda, text=instruccion).pack(anchor=tk.W, padx=10, pady=2)
        
        # Inicializar variables
        self.cap = None
        
        # Intentar cargar el modelo mejorado automáticamente al iniciar
        self.intentar_cargar_modelo_mejorado()
        
        # Iniciar la actualización de la interfaz
        self.actualizar_interfaz()
    
    def configurar_estilo(self):
        """Configura el estilo de la interfaz"""
        estilo = ttk.Style()
        estilo.configure("TFrame", background="#f0f0f0")
        estilo.configure("Card.TFrame", background="#ffffff", relief="raised")
        estilo.configure("Title.TLabel", font=("Helvetica", 12, "bold"))
        estilo.configure("TButton", font=("Helvetica", 10))
        estilo.configure("TLabelframe", background="#ffffff")
        estilo.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"))
    
    def intentar_cargar_modelo_mejorado(self):
        """Intenta cargar el modelo mejorado al iniciar la aplicación"""
        modelo_path = 'modelo_lenguaje_senhas_mejorado.h5'
        if os.path.exists(modelo_path):
            self.cargar_modelo_especifico(modelo_path)
        else:
            print(f"No se encontró el modelo mejorado en {modelo_path}")
    
    def cargar_modelo(self):
        """Carga un modelo seleccionado por el usuario"""
        ruta_modelo = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[("Archivos H5", "*.h5"), ("Todos los archivos", "*.*")]
        )
        
        if not ruta_modelo:
            return
        
        self.cargar_modelo_especifico(ruta_modelo)
    
    def cargar_modelo_mejorado(self):
        """Carga específicamente el modelo mejorado"""
        modelo_path = 'modelo_lenguaje_senhas_mejorado.h5'
        if os.path.exists(modelo_path):
            self.cargar_modelo_especifico(modelo_path)
        else:
            messagebox.showerror(
                "Error", 
                f"No se encontró el modelo mejorado en {modelo_path}.\n"
                "Primero ejecuta 'modelo_lenguaje_senhas_mejorado.py' para crear el modelo."
            )
    
    def cargar_modelo_especifico(self, ruta_modelo):
        """Carga un modelo desde una ruta específica"""
        global modelo
        
        try:
            self.estado_modelo_label.config(text="Cargando modelo...", foreground="orange")
            self.ventana.update()
            
            modelo = load_model(ruta_modelo)
            
            # Mostrar resumen del modelo
            print("Estructura del modelo cargado:")
            modelo.summary()
            
            self.estado_modelo_label.config(
                text=f"Estado: Modelo cargado desde {os.path.basename(ruta_modelo)}", 
                foreground="green"
            )
            self.btn_iniciar_captura.config(state=tk.NORMAL)
        except Exception as e:
            self.estado_modelo_label.config(text=f"Error al cargar modelo: {str(e)}", foreground="red")
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")
    
    def alternar_captura(self):
        """Inicia o detiene la captura de video"""
        global captura_activa, thread_captura
        
        if captura_activa:
            # Detener captura
            captura_activa = False
            self.btn_iniciar_captura.config(text="Iniciar Captura")
            if self.cap is not None:
                self.cap.release()
            self.btn_guardar_captura.config(state=tk.DISABLED)
        else:
            # Iniciar captura
            if modelo is None:
                self.estado_modelo_label.config(text="Primero debes cargar un modelo", foreground="red")
                return
            
            self.btn_iniciar_captura.config(text="Detener Captura")
            captura_activa = True
            self.btn_guardar_captura.config(state=tk.NORMAL)
            
            if thread_captura is None or not thread_captura.is_alive():
                thread_captura = threading.Thread(target=self.procesar_video)
                thread_captura.daemon = True
                thread_captura.start()
    
    def procesar_video(self):
        """Procesa el video de la webcam"""
        global prediccion_actual, confianza_actual, imagen_procesada, captura_activa, historial_predicciones
        
        # Inicializar la captura de video
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.estado_modelo_label.config(text="No se pudo abrir la cámara", foreground="red")
            captura_activa = False
            self.btn_iniciar_captura.config(text="Iniciar Captura")
            return
        
        ultima_prediccion = None
        contador_estable = 0
        tiempo_ultima_pred = 0
        
        while captura_activa:
            # Leer frame
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Voltear horizontalmente para efecto espejo
            frame = cv2.flip(frame, 1)
            
            # Convertir a RGB para MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Procesar el frame para detectar manos
            results = hands.process(frame_rgb)
            
            altura, anchura, _ = frame.shape
            region_mano = None
            hay_mano = False
            
            # Dibujar landmarks de las manos si están presentes
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if self.mostrar_landmarks.get():
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    
                    # Extraer región de la mano con margen adicional
                    puntos_x = [landmark.x for landmark in hand_landmarks.landmark]
                    puntos_y = [landmark.y for landmark in hand_landmarks.landmark]
                    
                    # Calcular el centro de la mano
                    centro_x = sum(puntos_x) / len(puntos_x)
                    centro_y = sum(puntos_y) / len(puntos_y)
                    
                    # Calcular la distancia máxima desde el centro para determinar el tamaño de la mano
                    max_dist = max(
                        max([abs(punto - centro_x) for punto in puntos_x]),
                        max([abs(punto - centro_y) for punto in puntos_y])
                    )
                    
                    # Aumentar el margen para capturar toda la mano
                    margen = max_dist * 1.5
                    
                    x_min = max(0, int((centro_x - margen) * anchura))
                    y_min = max(0, int((centro_y - margen) * altura))
                    x_max = min(anchura, int((centro_x + margen) * anchura))
                    y_max = min(altura, int((centro_y + margen) * altura))
                    
                    # Dibujar rectángulo alrededor de la mano
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    
                    # Extraer la región de la mano
                    if x_max > x_min and y_max > y_min:
                        region_mano = frame_rgb[y_min:y_max, x_min:x_max]
                        hay_mano = True
            
            # Solo procesar si hay una mano detectada o si no está activada la opción
            if hay_mano or not self.procesar_solo_mano.get():
                tiempo_actual = time.time()
                
                # Solo hacer predicción cada 0.3 segundos para mejor fluidez
                if tiempo_actual - tiempo_ultima_pred > 0.3:
                    if hay_mano and region_mano is not None:
                        # Procesar la imagen de la mano para la predicción
                        img_procesada = cv2.cvtColor(region_mano, cv2.COLOR_RGB2GRAY)
                        img_procesada = cv2.resize(img_procesada, (28, 28))
                        
                        # Normalizar y reshapear para el modelo
                        img_modelo = img_procesada.reshape(1, 28, 28, 1).astype('float32') / 255.0
                        
                        # Hacer predicción
                        prediccion = modelo.predict(img_modelo, verbose=0)[0]
                        indice_letra = np.argmax(prediccion)
                        
                        # Asegurarse de que el índice está dentro del rango válido
                        if 0 <= indice_letra < len(letras):
                            letra = letras[indice_letra]
                            confianza = float(prediccion[indice_letra]) * 100
                            
                            # Obtener umbral de confianza del slider
                            umbral = self.umbral_confianza.get()
                            
                            # Actualizar predicción solo si la confianza supera el umbral
                            if confianza > umbral:
                                prediccion_actual = letra
                                confianza_actual = confianza
                                
                                # Verificar si es la misma predicción que antes
                                if letra == ultima_prediccion:
                                    contador_estable += 1
                                else:
                                    contador_estable = 0
                                    ultima_prediccion = letra
                                
                                # Si la predicción es estable por 3 frames, agregarla al historial
                                if contador_estable == 3:
                                    tiempo_str = time.strftime("%H:%M:%S")
                                    historial_predicciones.append(f"{tiempo_str} - Letra: {letra} ({confianza:.2f}%)")
                                    contador_estable = 0
                                    
                                    # Limitar el historial a las últimas 50 predicciones
                                    if len(historial_predicciones) > 50:
                                        historial_predicciones = historial_predicciones[-50:]
                        
                        # Mostrar la letra y confianza en el frame
                        if prediccion_actual is not None:
                            cv2.putText(frame, f"Letra: {prediccion_actual}", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"Confianza: {confianza_actual:.2f}%", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Visualizar las 3 predicciones principales
                        top_indices = np.argsort(prediccion)[-3:][::-1]
                        y_pos = 110
                        for i, idx in enumerate(top_indices):
                            # Verificar que el índice es válido antes de usarlo
                            if 0 <= idx < len(letras):
                                conf = prediccion[idx] * 100
                                cv2.putText(frame, f"#{i+1}: {letras[idx]} ({conf:.1f}%)", (10, y_pos), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
                                y_pos += 30
                    
                    tiempo_ultima_pred = tiempo_actual
            
            # Dibujar información adicional en el frame
            cv2.putText(frame, f"Umbral: {self.umbral_confianza.get():.1f}%", (10, altura - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)
            
            if not hay_mano:
                cv2.putText(frame, "No se detecta mano", (anchura//2 - 120, altura//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Convertir el frame para mostrar en la interfaz
            imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagen_procesada = Image.fromarray(imagen_rgb)
        
        # Liberar la cámara
        self.cap.release()
    
    def actualizar_interfaz(self):
        """Actualiza la interfaz con la información actual"""
        global imagen_procesada, prediccion_actual, confianza_actual, historial_predicciones
        
        # Actualizar el panel de video
        if imagen_procesada is not None:
            # Calcular proporción para mantener relación de aspecto
            ancho_panel = self.panel_video.winfo_width()
            alto_panel = self.panel_video.winfo_height()
            
            if ancho_panel > 1 and alto_panel > 1:  # Asegurarse de que el panel tenga tamaño
                img_ancho, img_alto = imagen_procesada.size
                proporcion = min(ancho_panel/img_ancho, alto_panel/img_alto)
                nuevo_ancho = int(img_ancho * proporcion)
                nuevo_alto = int(img_alto * proporcion)
                
                imagen_redimensionada = imagen_procesada.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)
                imagen_tk = ImageTk.PhotoImage(imagen_redimensionada)
                self.panel_video.configure(image=imagen_tk)
                self.panel_video.image = imagen_tk
        
        # Actualizar la predicción
        if prediccion_actual is not None:
            self.letra_predicha.config(text=prediccion_actual)
            self.confianza_label.config(text=f"Confianza: {confianza_actual:.2f}%")
            
            # Cambiar el color según la confianza
            if confianza_actual > 90:
                self.letra_predicha.config(foreground="green")
            elif confianza_actual > 75:
                self.letra_predicha.config(foreground="orange")
            else:
                self.letra_predicha.config(foreground="red")
        
        # Actualizar historial
        if historial_predicciones:
            self.historial_text.config(state=tk.NORMAL)
            self.historial_text.delete(1.0, tk.END)
            
            # Mostrar las últimas 5 predicciones
            for pred in historial_predicciones[-5:]:
                self.historial_text.insert(tk.END, pred + "\n")
            
            self.historial_text.config(state=tk.DISABLED)
        
        # Programar próxima actualización
        self.ventana.after(10, self.actualizar_interfaz)
    
    def limpiar_historial(self):
        """Limpia el historial de predicciones"""
        global historial_predicciones
        historial_predicciones = []
        self.historial_text.config(state=tk.NORMAL)
        self.historial_text.delete(1.0, tk.END)
        self.historial_text.config(state=tk.DISABLED)
    
    def guardar_captura(self):
        """Guarda la imagen actual como archivo"""
        global imagen_procesada
        
        if imagen_procesada is None:
            messagebox.showinfo("Información", "No hay imagen para guardar")
            return
        
        # Obtener la ruta del archivo para guardar
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos los archivos", "*.*")]
        )
        
        if filename:
            try:
                imagen_procesada.save(filename)
                messagebox.showinfo("Éxito", f"Imagen guardada como {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar la imagen: {str(e)}")
    
    def on_closing(self):
        """Manejador para el cierre de la ventana"""
        global captura_activa
        captura_activa = False
        if self.cap is not None:
            self.cap.release()
        self.ventana.destroy()

def main():
    """Función principal"""
    # Verificar si existe el modelo mejorado
    modelo_mejorado_path = 'modelo_lenguaje_senhas_mejorado.h5'
    if not os.path.exists(modelo_mejorado_path):
        # Buscar en directorio checkpoints
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            modelos = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.h5')]
            if modelos:
                # Ordenar por precisión (valor después de val_acc_ en el nombre)
                modelos.sort(key=lambda x: float(x.split('val_acc_')[1].split('.h5')[0]), reverse=True)
                mejor_modelo = modelos[0]
                print(f"Se encontró un modelo entrenado en checkpoints: {mejor_modelo}")
                # Copiarlo a la ubicación estándar
                import shutil
                shutil.copy(mejor_modelo, modelo_mejorado_path)
                print(f"Copiado como {modelo_mejorado_path}")
            else:
                print(f"AVISO: No se encontró el modelo mejorado en {modelo_mejorado_path}")
                print("Se recomienda ejecutar primero 'modelo_lenguaje_senhas_mejorado.py' para crear el modelo optimizado.")
        else:
            print(f"AVISO: No se encontró el modelo mejorado en {modelo_mejorado_path}")
            print("Se recomienda ejecutar primero 'modelo_lenguaje_senhas_mejorado.py' para crear el modelo optimizado.")
    
    # Crear la interfaz gráfica
    root = tk.Tk()
    app = AplicacionLenguajeSenhas(root)
    
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main() 