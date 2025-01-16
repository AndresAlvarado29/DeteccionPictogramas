from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import cv2

app = FastAPI()

# Configurar la carpeta de templates
templates = Jinja2Templates(directory="templates")

# Iniciar la captura de video
camera = cv2.VideoCapture(0)

@app.get("/")
def index(request: Request):
    # Renderizar el archivo index.html
    return templates.TemplateResponse("index.html", {"request": request})

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Codificar el frame como un flujo de bytes JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Cierra la cámara cuando se detiene el servidor
@app.on_event("shutdown")
def shutdown_event():
    camera.release()
