# Librerías
import torch
import cv2
import numpy as np
from imutils.video import VideoStream
from midas.model_loader import default_models, load_model
import utils

# Desactiva el cálculo de gradientes
@torch.no_grad()

############ CÁLCULO DE MATRIZ DE PROFUNDIDAD ############

# Función que llevará a cabo el proceso:
            #CPU o GPU, modelo de red, matriz imagen, 
                                #tamaño de entrada, tamaño buscado
def process(device, model, image, input_size, target_size):

    # Convierte la imagen (image) a un tensor de PyTorch y lo envía a CPU o GPU.
    sample = torch.from_numpy(image).to(device).unsqueeze(0)
    # Procesa la imagen y devuelve la matriz de profundidad estimada.
    prediction = model.forward(sample)
    # Utiliza la función de PyTorch 'interpolate' para cambiar el tamaño de la matriz de
    # profundidad estimada al tamaño destino.
    prediction = (
    
        torch.nn.functional.interpolate(
            # Agrega una dimensión adicional a la matriz de profundidad
            prediction.unsqueeze(1),
            # Tamaño de destino de la matriz de profundidad.
            # [..-1 ] se usa para invertir el orden de los elementos de la tupla.
            size=target_size[::-1],
            # Modo de interpolación utilizado para cambiar el tamaño
            # de la matriz de profundidad estimada.
            mode="bicubic",
            # ¿ Las esquinas de los pixeles deben alinearse durante la interpolación ?
            align_corners= False,
        )

        # Elimina las dimensiones adicionales de la matriz
        # de profundidad y la convierte en arreglo NumPy.
        .squeeze()
        .cpu()
        .numpy()
    )

    # Regresa la matriz de profundidad estimada.
    return prediction

####### IMAGEN ORIGINAL Y MAPA DE PROFUNDIDAD ###########

def create_side_by_side(image, depth):
    
    # Utiliza las funciones de NumPy para obetener el
    # valor mínimo
    depth_min = depth.min()
    # y máximo de la matriz de profundidad
    depth_max = depth.max()

    # Escala los valores de la matriz al rango [0,1] y luego
    # Los multiplica para obtener de [0, 255].
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min) 
    # Multiplica la matriz de profundidad normalizada por 3 para aumentar el contraste.
    normalized_depth *= 3

    # expand_dims: Agrega una dimensión a la matriz de 
                #  profundidad normalizada. (array, axis position)
    # repeat: Repite un elemento de la matriz justo después de este.
                # ( matriz, repeticiones, axis position)

    # Repite la matriz de profundidad normalizada a lo largo de la dimensión 
    # 2 para crear una imagen en color, la divide sobre tres para
    # reducir la intensidad del color.
    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    
    # Aplica el mapa de colores para resaltar profundidad
                                    # Cast a uint 8.
    right_side = cv2.applyColorMap (np.uint8(right_side), cv2.COLORMAP_INFERNO)
    
    # Junta la imagen y la matriz colorizada llamada right_side.
    return np.concatenate ((image, right_side), axis= 1)   

############ CARGAR MODELO ############

def run():
    # Determina si correrá en CPU o GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ¿Modelo optimizado para mejorar rendimiento? < Accurate.
    optimize = False
    # ¿Muestra las imágenes de cada lado de manera predeterminada?
    side = False
    # Altura de la imagen de entrada, None la ajusta automáticamente.
    height= None
    # ¿Se ajusta a cuadrado? Sino, se ajustará a rectángulo.
    square= False
    # La imagen de entrada se establece en escala de grises.
    grayscale = False
    # Modelo usado, recuerda que los pesos en .pt deben estar cargados en
    # la carpeta weights del modelo.
    model_type = "dpt_swin2_tiny_256"
    # Carga el modelo especificado a partir de la carpeta de los pesos.
    model_path = default_models[model_type]
    # Carga:
    # Modelo, Transformador, Ancho de red y altura de red.
    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize, height, square)

    # Comienza el video a partir de la fuente 0.
    video = VideoStream(0).start()

    while True:
        # Hace la lectura del frame actual del video.
        frame = video.read()

        # Condicional, si existe frame entonces:
        if frame is not None:
            # Convierte el cuadro en imagen RGB (0, 255).
            original_image_rgb = np.flip(frame, 2)
            # Transforma y normaliza la imagen RGB:
                # Transform viene de la biblioteca MiDaS.
                            # Normaliza la imagen (0,1).
                                # ["image"] es la clave del diccionario de la
                                # variable transformada.
            image = transform ({"image": original_image_rgb/255})["image"]

            # Llama la función process del modelo para hacer la estimación
            prediction = process(device, model, image, (net_w, net_h), original_image_rgb.shape[1::-1])
            # Convierte el frame en RGB
            original_image_bgr = np.flip(original_image_rgb, 2)
            # Toma el frame y realiza la función create side_by_side
            content = create_side_by_side(original_image_bgr, prediction)
            # Muestra las imágenes juntas normalizadas
            cv2.imshow('MiDaS Depth Estimation - Press Esc to close the Window', content/255)
            # Si la tecla 27 (Esc) es presionada, entonces se cierra la transmisión y se rompe el proceso
            if cv2.waitKey(1) == 27:
                break 

    # Cierra las ventanas y termina la filmación.
    cv2.destroyAllWindows()
    video.stop()

# Si main es main, entonces ejecuta run

if __name__ == '__main__':
    run()
