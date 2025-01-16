from TTS.api import TTS

# Configurar el modelo
# Puedes cambiar "tts_models/multilingual/multi-dataset/xtts_v2" por otro modelo compatible
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Texto de entrada para la síntesis
text = "Hola, esta es una prueba de conversión de texto a voz usando Coqui y quiero ver si esto me ayuda para mi tesis ya que quiero que esto se use cuando se haga la prediccion del pictograma"

# Ruta al archivo de voz objetivo (para clonación de voz)
# Reemplaza con la ruta a tu archivo de audio de referencia
speaker_wav = "voz/miVoz.wav"

# Generar audio
output_path = "voz/output.wav"
tts.tts_to_file(
    text=text,
    file_path=output_path,
    speaker_wav=speaker_wav,  # Clonación de voz
    language="es"             # Idioma: Cambiar a "en" para inglés
)

print(f"Audio generado guardado en: {output_path}")
