import torch
import cv2
import numpy as np

# Lista de nombres de clases
class_names = [
    'doctor', 'mecanico', 'domestico', 'salvaje', 'noche', 
    'dia', 'montaña', 'playa', 'granja', 'dulce', 'saludable'
]

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Cargar modelo TorchScript
model = torch.jit.load('./model/best9.torchscript')
model = model.to(device)

# Configurar captura de video desde la cámara
cap = cv2.VideoCapture(0)  # 0 es el ID de la cámara por defecto

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

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

    # Dibujar detecciones en el frame original
    for box, conf, cls in zip(filtered_boxes, filtered_confidences, filtered_classes):
        # Denormalizar las coordenadas del bounding box
        x1, y1, x2, y2 = box
        x1 = int(x1 * original_width / 640)
        y1 = int(y1 * original_height / 640)
        x2 = int(x2 * original_width / 640)
        y2 = int(y2 * original_height / 640)

        # Obtener el nombre de la clase
        class_name = class_names[int(cls)]
        label = f"{class_name} {conf:.2f}"

        # Dibujar el rectángulo y el texto
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
