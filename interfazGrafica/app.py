from flask import Flask, render_template, Response, request
import cv2
import torch
import soundfile as sf
from transformers import pipeline
from ultralytics import YOLO
from pathlib import Path

# Inicializar Flask
app = Flask(__name__)

# Configurar la cámara
camera = cv2.VideoCapture(0)

# Cargar el modelo YOLO (torchscript o TensorRT engine)
model_path = "/ruta/a/tu/modelo/best.engine"  # Cambia a .torchscript si es necesario
model = torch.jit.load(model_path) if model_path.endswith(".torchscript") else YOLO(model_path)

# Inicializar el modelo de text-to-speech
tts_pipeline = pipeline("text-to-speech", model="coqui/XTTS-v2")

# Función para capturar la cámara y mostrarla en tiempo real
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Convertir a formato JPEG para mostrar en la página web
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Ruta para mostrar el feed de la cámara
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta principal para la página web
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar una imagen y devolver la predicción
@app.route('/predict', methods=['POST'])
def predict():
    # Leer la imagen subida desde el formulario
    file = request.files['image']
    img_path = "uploaded_image.jpg"
    file.save(img_path)

    # Realizar predicción con YOLO
    results = model(img_path)  # Cambiar lógica si usas .engine o TorchScript
    predictions = []
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            predictions.append((cls, box.tolist()))

    # Crear texto con las predicciones
    prediction_text = ", ".join([f"{model.names[int(cls)]}" for cls, _ in predictions])

    # Usar el modelo TTS para generar el audio
    audio = tts_pipeline(prediction_text)
    audio_path = "prediction_audio.wav"
    sf.write(audio_path, audio["audio"], samplerate=audio["sampling_rate"])

    return {"predictions": predictions, "audio": audio_path}

# Ruta para descargar el audio generado
@app.route('/download_audio')
def download_audio():
    return Response(open("prediction_audio.wav", "rb"), mimetype="audio/wav")

if __name__ == '__main__':
    app.run(debug=True)
