import os
import sys
import glob
import shutil

def buscar_mejor_modelo():
    """Busca el mejor modelo disponible y lo copia como modelo principal si es necesario"""
    modelo_principal = 'modelo_lenguaje_senhas_mejorado.h5'
    
    # 1. Comprobar si el modelo principal ya existe
    if os.path.exists(modelo_principal):
        print(f"Modelo principal encontrado: {modelo_principal}")
        return modelo_principal
    
    # 2. Buscar en la carpeta de checkpoints
    checkpoint_dir = "checkpoints"
    if os.path.exists(checkpoint_dir):
        modelos = glob.glob(os.path.join(checkpoint_dir, "*.h5"))
        if modelos:
            # Ordenar por precisión (suponiendo nombre modelo_senhas_epoch_XX_val_acc_0.XXXX.h5)
            modelos.sort(key=lambda x: float(x.split('val_acc_')[1].split('.h5')[0]) 
                          if 'val_acc_' in x else 0, 
                          reverse=True)
            mejor_modelo = modelos[0]
            print(f"Se usará el mejor modelo encontrado: {mejor_modelo}")
            
            # Copiar al modelo principal
            shutil.copy(mejor_modelo, modelo_principal)
            print(f"Copiado como {modelo_principal}")
            return modelo_principal
    
    # 3. Buscar cualquier modelo .h5 en el directorio actual
    modelos_dir = glob.glob("*.h5")
    if modelos_dir:
        mejor_modelo = modelos_dir[0]
        print(f"Se usará el modelo encontrado: {mejor_modelo}")
        return mejor_modelo
    
    print("No se encontró ningún modelo entrenado.")
    return None

def ejecutar_interfaz(modelo=None):
    """Ejecuta la interfaz gráfica con el modelo especificado"""
    try:
        print("Iniciando la interfaz gráfica...")
        from deteccion_senhas_gui import AplicacionLenguajeSenhas
        import tkinter as tk
        
        # Crear la interfaz gráfica
        root = tk.Tk()
        app = AplicacionLenguajeSenhas(root)
        
        # Cargar el modelo automáticamente si se especificó
        if modelo and os.path.exists(modelo):
            print(f"Cargando automáticamente el modelo: {modelo}")
            app.cargar_modelo_especifico(modelo)
        
        # Configurar el cierre de la ventana
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
        
    except ImportError as e:
        print(f"Error al importar módulos: {e}")
        print("Asegúrate de tener todas las dependencias instaladas:")
        print("pip install -r requirements.txt")
    except Exception as e:
        print(f"Error al iniciar la interfaz: {e}")

if __name__ == "__main__":
    # Buscar el mejor modelo disponible
    modelo = buscar_mejor_modelo()
    
    # Ejecutar la interfaz
    ejecutar_interfaz(modelo) 