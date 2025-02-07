import torch
import cv2
from TTS.api import TTS
import os
import time  # Para gestionar el delay

# Lista de nombres de clases
class_names = [
    'doctor', 'mecanico', 'domestico', 'salvaje', 'noche', 
    'dia', 'montaña', 'playa', 'granja', 'dulce', 'saludable'
]

# Inicializar el modelo TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar modelo TorchScript
model = torch.jit.load('./model/newBest4.torchscript')
model = model.to(device)
model.eval()

# Configurar captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara por defecto

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Diccionario para almacenar el último tiempo de reproducción por clase
last_played = {class_name: 0 for class_name in class_names}
delay_time = 7  # Delay en segundos entre reproducciones para la misma clase

# Procesamiento en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar el frame.")
        break

    # Obtener dimensiones originales del frame
    original_height, original_width = frame.shape[:2]

    # Preprocesar el frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (640, 640))
    frame_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame_tensor = frame_tensor.to(device)

    # Realizar inferencia
    with torch.no_grad():
        results = model(frame_tensor)[0]

    # Procesar resultados
    detections = results[0].cpu().numpy()  # [25200, 16]

    # Extraer valores útiles
    boxes = detections[:, :4]
    confidences = detections[:, 4]
    class_probs = detections[:, 5:]

    # Filtrar detecciones
    confidence_threshold = 0.5
    indices = confidences > confidence_threshold

    filtered_boxes = boxes[indices]
    filtered_confidences = confidences[indices]
    filtered_classes = class_probs[indices].argmax(axis=1)

    # Dibujar detecciones en el frame original y mostrar en consola
    for box, conf, cls in zip(filtered_boxes, filtered_confidences, filtered_classes):
        # Denormalizar las coordenadas del bounding box
        x1, y1, x2, y2 = box
        x1 = int(x1 * original_width / 640)
        y1 = int(y1 * original_height / 640)
        x2 = int(x2 * original_width / 640)
        y2 = int(y2 * original_height / 640)

        # Obtener el nombre de la clase
        class_name = class_names[int(cls)]

        # Imprimir predicción en consola
        print(f"Predicción: {class_name}, Confianza: {conf:.2f}, Coordenadas: ({x1}, {y1}), ({x2}, {y2})")

        # Reproducir audio si ha pasado el tiempo del delay
        current_time = time.time()
        if current_time - last_played[class_name] > delay_time:
            speaker_wav = "voz/vozAndrea.wav"
            # Generar audio para la predicción
            audio_output = f"voz/{class_name}.wav"
            tts.tts_to_file(
                text=f"Se detectó {class_name}",
                speaker_wav=speaker_wav,
                file_path=audio_output,
                language="es"
            )

            # Reproducir el audio generado
            os.system(f"aplay {audio_output}")

            # Actualizar el tiempo de reproducción
            last_played[class_name] = current_time

        # Dibujar el rectángulo y el texto
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame con las detecciones
    cv2.imshow("Detections", frame)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()