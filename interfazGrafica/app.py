import cv2
import torch
import numpy as np
import time
import os
from TTS.api import TTS

# Clases del modelo
classes = ['doctor', 'mecanico', 'domestico', 'salvaje', 'noche', 'dia', 
           'montaña', 'playa', 'granja', 'dulce', 'saludable']

# Colores para los bounding boxes (un color diferente para cada clase)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Inicializar TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Diccionario para almacenar el último tiempo de reproducción por clase
last_played = {class_name: 0 for class_name in classes}
delay_time = 7  # Delay en segundos entre reproducciones para la misma clase

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)

def load_model(model_path):
    # Cargar modelo TorchScript
    model = torch.jit.load(model_path)
    # Mover el modelo a GPU si está disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    model.to(device)
    # Poner en modo evaluación
    model.eval()
    return model, device

def detect(model, img, device, conf_thres=0.25, iou_thres=0.45):
    # Preprocesar la imagen
    img_input = img.copy()
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)
    img_input, ratio, pad = letterbox(img_input, 640, auto=False)
    img_input = img_input.transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
    img_input = np.ascontiguousarray(img_input)
    img_input = torch.from_numpy(img_input).to(device)
    img_input = img_input.float()  # uint8 to fp16/32
    img_input /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_input.ndimension() == 3:
        img_input = img_input.unsqueeze(0)
    
    # Inferencia
    with torch.no_grad():
        pred = model(img_input)
        if isinstance(pred, tuple):
            pred = pred[0]  # Algunos modelos retornan tuple
        
    # Aplicar Non-Maximum Suppression
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    return pred, ratio, pad

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Performs Non-Maximum Suppression (NMS) on inference results"""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision_nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        output[xi] = x[i]
    return output

def torchvision_nms(boxes, scores, iou_thres):
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_thres)
    except:
        # Fallback to regular NMS implementation
        boxes_cpu = boxes.cpu().numpy()
        scores_cpu = scores.cpu().numpy()
        
        # Simple NMS implementation
        indices = []
        while len(boxes_cpu) > 0:
            idx = np.argmax(scores_cpu)
            indices.append(idx)
            
            # Get IoU for all remaining boxes
            ious = bbox_iou(boxes_cpu[idx], boxes_cpu)
            mask = ious < iou_thres
            
            # Remove boxes with high IoU
            boxes_cpu = boxes_cpu[mask]
            scores_cpu = scores_cpu[mask]
        
        return torch.tensor(indices, device=boxes.device)

def bbox_iou(box1, boxes):
    # Calculate intersection area
    x1 = np.maximum(box1[0], boxes[:, 0])
    y1 = np.maximum(box1[1], boxes[:, 1])
    x2 = np.minimum(box1[2], boxes[:, 2])
    y2 = np.minimum(box1[3], boxes[:, 3])
    
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersection = w * h
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box1_area + boxes_area - intersection
    
    # Calculate IoU
    iou = intersection / (union + 1e-16)
    return iou

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def scale_coords(coords, img_shape):
    # Ajustar las coordenadas detectadas al tamaño original de la imagen
    img_h, img_w = img_shape[:2]
    
    # Determinar si estamos trabajando con tensor o numpy array
    is_tensor = isinstance(coords, torch.Tensor)
    
    if is_tensor:
        # Clone coords si es un tensor
        scaled_coords = coords.clone()
        device = coords.device
        
        # Normalize coordinates to [0, 1]
        scaled_coords[:, [0, 2]] /= 640
        scaled_coords[:, [1, 3]] /= 640
        
        # Scale to image size
        scaled_coords[:, [0, 2]] *= img_w
        scaled_coords[:, [1, 3]] *= img_h
        
        # Ensure coordinates are within image boundaries
        scaled_coords[:, [0, 2]] = torch.clamp(scaled_coords[:, [0, 2]], 0, img_w)
        scaled_coords[:, [1, 3]] = torch.clamp(scaled_coords[:, [1, 3]], 0, img_h)
        
    else:
        # Clone coords si es un numpy array
        scaled_coords = coords.copy()
        
        # Normalize coordinates to [0, 1]
        scaled_coords[:, [0, 2]] /= 640
        scaled_coords[:, [1, 3]] /= 640
        
        # Scale to image size
        scaled_coords[:, [0, 2]] *= img_w
        scaled_coords[:, [1, 3]] *= img_h
        
        # Ensure coordinates are within image boundaries
        scaled_coords[:, [0, 2]] = np.clip(scaled_coords[:, [0, 2]], 0, img_w)
        scaled_coords[:, [1, 3]] = np.clip(scaled_coords[:, [1, 3]], 0, img_h)
    
    return scaled_coords

def plot_one_box(x, img, color, label=None, line_thickness=None):
    # Dibujar un bounding box en la imagen
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def play_tts(class_name, speaker_wav="voz/vozAndrea.wav"):
    """Reproduce la detección usando TTS"""
    current_time = time.time()
    
    # Verificar si ha pasado suficiente tiempo desde la última reproducción
    if current_time - last_played[class_name] > delay_time:
        # Crear directorio para los archivos de audio si no existe
        os.makedirs("voz", exist_ok=True)
        
        # Generar audio para la predicción
        audio_output = f"voz/{class_name}.wav"
        
        # Generar el audio solo si no existe o forzar regeneración
        if not os.path.exists(audio_output):
            print(f"Generando audio para: {class_name}")
            tts.tts_to_file(
                text=f"Se detectó {class_name}",
                speaker_wav=speaker_wav,
                file_path=audio_output,
                language="es"
            )
        
        # Reproducir el audio generado
        print(f"Reproduciendo audio para: {class_name}")
        os.system(f"aplay {audio_output}")
        
        # Actualizar el tiempo de reproducción
        last_played[class_name] = current_time
        return True
    
    return False

def main():
    # Ruta al modelo TorchScript (cambia esto a la ubicación de tu modelo)
    model_path = './model/newBest4.torchscript'
    
    # Cargar el modelo
    try:
        model, device = load_model(model_path)
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Iniciar la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la webcam")
        return
    
    # Preparar directorio para archivos de voz
    os.makedirs("voz", exist_ok=True)
    speaker_wav = "voz/vozAndrea.wav"
    
    # Verificar si existe el archivo de voz del speaker
    if not os.path.exists(speaker_wav):
        print(f"Advertencia: Archivo {speaker_wav} no encontrado. La TTS podría no funcionar correctamente.")
    
    print("Presiona 'q' para salir")
    
    while True:
        # Leer frame de la webcam
        ret, frame = cap.read()
        if not ret:
            print("Error al leer frame de la webcam")
            break
        
        # Obtener dimensiones originales
        original_height, original_width = frame.shape[:2]
        
        # Realizar detección
        start_time = time.time()
        detections, ratio, pad = detect(model, frame, device, conf_thres=0.4)
        fps = 1.0 / (time.time() - start_time)
        
        # Procesar detecciones
        for det in detections:
            if len(det):
                # Redimensionar las coordenadas a la imagen original
                det_scaled = det.clone()
                det_scaled[:, :4] = scale_coords(det[:, :4], frame.shape)
                
                # Dibujar bounding boxes
                for *xyxy, conf, cls_idx in det_scaled.cpu().numpy():
                    cls_idx = int(cls_idx)
                    class_name = classes[cls_idx]
                    label = f"{class_name} {conf:.2f}"
                    
                    # Imprimir predicción en consola
                    print(f"Predicción: {class_name}, Confianza: {conf:.2f}, Coordenadas: ({xyxy[0]:.1f}, {xyxy[1]:.1f}), ({xyxy[2]:.1f}, {xyxy[3]:.1f})")
                    
                    # Reproducir audio TTS
                    play_tts(class_name, speaker_wav)
                    
                    # Dibujar bounding box
                    plot_one_box(xyxy, frame, label=label, color=colors[cls_idx])
        
        # Mostrar FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Mostrar frame
        cv2.imshow('YOLOv5 TorchScript Detector con TTS', frame)
        
        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        import torchvision
        print("Torchvision encontrado, usando NMS optimizado.")
    except ImportError:
        print("Torchvision no está instalado. Usando implementación fallback para NMS.")
    
    main()