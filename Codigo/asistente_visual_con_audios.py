import os
import sys
import time
import platform

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from audio_player import AudioNotifier

# --------------------------------------------------------------------------
# CONFIGURACIÓN
# --------------------------------------------------------------------------
IDIOMA = "es"                 # "es" | "en" | "gn"  -> podés cambiarlo en caliente con las teclas 1/2/3
COOLDOWN_AUDIO = 2.5            # segundos entre anuncios (ahora es UN cooldown global, no por tercio)
DEPTH_EVERY_N_FRAMES = 2        # MiDaS es el modelo más pesado, no hace falta correrlo cada frame
RESIZE_WIDTH = 640              # 480 recortaba demasiado detalle para reconocer sillas/mesas/bancos.
                                 # Si tu PC no tiene GPU y se pone muy lento, bajalo a 480 o 416.
CONF_YOLO = 0.25                # confianza mínima para aceptar una detección (0.0 a 1.0).
                                 # Si sigue sin reconocer objetos, bajalo a 0.15-0.20 (aumenta falsos positivos).
UMBRAL_CERCA = 1.2
UMBRAL_MEDIA = 2.5

# Nombre de clase COCO (como lo devuelve YOLO) -> clave interna usada en audio_mapper.json
YOLO_A_CLAVE = {
    "person": "person",
    "chair": "chair",
    "bench": "bench",
    "dining table": "table",
}
CLASES_YOLO = list(YOLO_A_CLAVE.keys())

# Solo para lo que se dibuja en pantalla (no afecta al audio)
NOMBRE_VISUAL = {
    "person": "Persona", "chair": "Silla", "bench": "Banco", "table": "Mesa",
    "door": "Puerta", "stairs": "Escalera", "obstacle": "Obstáculo",
}

# --------------------------------------------------------------------------
# REPRODUCTOR DE AUDIO (pregrabado, no bloqueante gracias a pygame.mixer)
# --------------------------------------------------------------------------
notifier = AudioNotifier()  # busca audio_mapper.json en la misma carpeta que este script

# --------------------------------------------------------------------------
# INICIALIZACIÓN DE MODELOS
# --------------------------------------------------------------------------
torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model_yolo = YOLO("yolov8n.pt")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.to(device)
    midas.eval()
    if device.type == "cuda":
        midas.half()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
except Exception as e:
    print(f"Error iniciando modelos: {e}")
    sys.exit(1)

backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
cap = cv2.VideoCapture(0, backend)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    sys.exit(1)


def estimar_profundidad(frame_bgr):
    """Profundidad RELATIVA de MiDaS (no metros calibrados)."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    if device.type == "cuda":
        input_batch = input_batch.half()
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.float().cpu().numpy()


def distancia_a_bucket(dist_m):
    if dist_m < UMBRAL_CERCA:
        return "close", "MUY CERCA", (0, 0, 255)
    elif dist_m <= UMBRAL_MEDIA:
        return "media", "DISTANCIA MEDIA", (0, 255, 255)
    else:
        return "far", "LEJOS", (0, 255, 0)


ultimo_audio_time = 0.0  # cooldown GLOBAL: solo se anuncia una cosa a la vez, la más cercana
frame_idx = 0
depth_map = None

print("Presioná 'q' para salir. Teclas '1'=es '2'=en '3'=gn.")
print("Teclas de demo manual (para clases que YOLO no detecta): 'p'=puerta  'l'=escalera")

try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("No se pudo leer frame de la cámara, deteniendo.")
            break

        frame = cv2.flip(frame, 1)  # efecto espejo

        alto_orig, ancho_orig = frame.shape[:2]
        escala = RESIZE_WIDTH / ancho_orig
        if escala < 1.0:
            frame_proc = cv2.resize(frame, (RESIZE_WIDTH, int(alto_orig * escala)))
        else:
            frame_proc = frame

        alto, ancho = frame_proc.shape[:2]
        tercio = ancho // 3

        if frame_idx % DEPTH_EVERY_N_FRAMES == 0 or depth_map is None:
            depth_map = estimar_profundidad(frame_proc)
        frame_idx += 1

        results = model_yolo(
            frame_proc, stream=True, verbose=False, conf=CONF_YOLO,
            classes=[i for i, n in model_yolo.names.items() if n in CLASES_YOLO],
        )

        # candidatos por tercio: {"left": (dist, clave, estado, color) o None, ...}
        candidatos = {"left": None, "front": None, "right": None}

        def posicion_de(centro_x):
            if centro_x < tercio:
                return "left"
            elif centro_x > tercio * 2:
                return "right"
            return "front"

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                nombre_coco = model_yolo.names[cls]
                clave = YOLO_A_CLAVE.get(nombre_coco)
                if clave is None:
                    continue

                pos = posicion_de((x1 + x2) // 2)

                ancho_box, alto_box = x2 - x1, y2 - y1
                x_min_c = max(0, x1 + int(ancho_box * 0.3))
                x_max_c = min(ancho, x1 + int(ancho_box * 0.7))
                y_min_c = max(0, y1 + int(alto_box * 0.3))
                y_max_c = min(alto, y1 + int(alto_box * 0.7))
                roi = depth_map[y_min_c:y_max_c, x_min_c:x_max_c]
                if roi.size == 0:
                    continue

                profundidad_val = float(np.mean(roi))
                dist_m = max(0.4, min(6.0, 1000.0 / (profundidad_val + 1e-5)))
                bucket, estado, color = distancia_a_bucket(dist_m)

                cv2.rectangle(frame_proc, (x1, y1), (x2, y2), color, 2)
                texto = f"{NOMBRE_VISUAL[clave]} {pos}: {dist_m:.1f}m"
                cv2.putText(frame_proc, texto, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                actual = candidatos[pos]
                if actual is None or dist_m < actual[0]:
                    candidatos[pos] = (dist_m, clave, bucket, estado)

        # "Obstáculo" genérico: tercios donde YOLO no reconoció nada
        # conocido pero MiDaS igual detecta algo muy cerca.
        for pos, x_range in (
            ("left", (0, tercio)),
            ("front", (tercio, tercio * 2)),
            ("right", (tercio * 2, ancho)),
        ):
            if candidatos[pos] is not None:
                continue  # ya hay un objeto real identificado ahí
            franja = depth_map[:, x_range[0]:x_range[1]]
            if franja.size == 0:
                continue
            profundidad_val = float(np.mean(franja))
            dist_m = max(0.4, min(6.0, 1000.0 / (profundidad_val + 1e-5)))
            bucket, estado, color = distancia_a_bucket(dist_m)
            if bucket in ("close", "media"):
                candidatos[pos] = (dist_m, "obstacle", bucket, estado)

        # MODO PRIORIDAD: juntamos todo lo detectado en los 3 tercios y nos
        # quedamos únicamente con el más cercano para anunciar primero.
        alertas_validas = [
            (dist_m, pos, clave, bucket, estado)
            for pos, cand in candidatos.items()
            if cand is not None
            for dist_m, clave, bucket, estado in [cand]
            if bucket in ("close", "media")
        ]

        tiempo_actual = time.time()
        if alertas_validas and (tiempo_actual - ultimo_audio_time > COOLDOWN_AUDIO):
            alertas_validas.sort(key=lambda a: a[0])  # más cerca primero
            dist_m, pos, clave, bucket, estado = alertas_validas[0]
            notifier.play_warning(obj_class=clave, position=pos, distance=bucket, lang=IDIOMA)
            ultimo_audio_time = tiempo_actual

        cv2.line(frame_proc, (tercio, 0), (tercio, alto), (255, 255, 255), 1)
        cv2.line(frame_proc, (tercio * 2, 0), (tercio * 2, alto), (255, 255, 255), 1)
        cv2.putText(frame_proc, f"Idioma: {IDIOMA}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Asistente para Personas Ciegas", frame_proc)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        elif tecla == ord("1"):
            IDIOMA = "es"
        elif tecla == ord("2"):
            IDIOMA = "en"
        elif tecla == ord("3"):
            IDIOMA = "gn"
        elif tecla == ord("p"):
            # Demo manual: YOLOv8n de base no reconoce "puerta" (no es clase COCO).
            notifier.play_warning(obj_class="door", position="front", distance="media", lang=IDIOMA)
        elif tecla == ord("l"):
            # Demo manual: YOLOv8n de base no reconoce "escalera" (no es clase COCO).
            notifier.play_warning(obj_class="stairs", position="front", distance="media", lang=IDIOMA)

finally:
    cap.release()
    cv2.destroyAllWindows()
