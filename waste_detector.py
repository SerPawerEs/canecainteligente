#!/usr/bin/env python3
"""
Detector de desechos / basura en tiempo real con OpenCV + YOLO.

- Usa por defecto YOLO-World (detección de vocabulario abierto): detecta
  bolsas plásticas, botellas, papel arrugado, latas, cartón, etc. SIN entrenar.
- Opcionalmente puedes usar tus propios pesos entrenados (--model mis_pesos.pt),
  por ejemplo entrenados con el dataset TACO u otro de residuos.
- Muestra la imagen con cada objeto señalado y las coordenadas de su centro,
  tomando el centro de la cámara como origen (0, 0):
      X positivo -> derecha     X negativo -> izquierda
      Y positivo -> arriba      Y negativo -> abajo

Controles:
    q / ESC : salir
    s       : guardar captura (carpeta capturas/)
    c       : imprimir coordenadas en consola (on/off)
    espacio : pausar / reanudar el carro (parada de emergencia)

Ejemplos:
    python waste_detector.py
    python waste_detector.py --camera 1 --conf 0.15
    python waste_detector.py --model runs/detect/train/weights/best.pt
    python waste_detector.py --px-per-cm 12.5      # coordenadas en cm

Control del carro (Arduino por USB o HC-06):
    python waste_detector.py --list-ports          # ver puertos seriales
    python waste_detector.py --port COM5           # mueve el carro hacia el objeto
    (sin --port solo se simula en pantalla lo que haría el carro)
    Reglas: Y>0 adelante | Y<0 atrás | X<0 izquierda | X>0 derecha
    Si el carro se mueve al revés usa --invert-x y/o --invert-y
"""

import argparse
import os
import platform
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from car_controller import CMD_NAMES, CarController, decide_command_box, list_serial_ports, speed_for

# Clases (en inglés porque el modelo de texto-imagen CLIP funciona mejor así).
# Se pueden editar libremente o pasar con --classes "a,b,c".
DEFAULT_CLASSES = [
    "plastic bag",
    "plastic bottle",
    "bottle",
    "water bottle",
    "soda bottle",
    "crumpled paper",
    "paper",
    "cardboard",
    "can",
    "aluminum can",
    "glass bottle",
    "food wrapper",
    "plastic cup",
    "disposable cup",
    "plastic container",
    "food waste",
    "banana peel",
    "styrofoam",
    "tissue",
    "cigarette butt",
    "trash",
    "garbage",
]

# Traducción para mostrar en pantalla (opcional)
ES = {
    "plastic bag": "bolsa plastica",
    "plastic bottle": "botella plastica",
    "bottle": "botella",
    "water bottle": "botella de agua",
    "soda bottle": "botella gaseosa",
    "crumpled paper": "papel arrugado",
    "paper": "papel",
    "cardboard": "carton",
    "can": "lata",
    "aluminum can": "lata aluminio",
    "glass bottle": "botella vidrio",
    "food wrapper": "empaque comida",
    "plastic cup": "vaso plastico",
    "disposable cup": "vaso desechable",
    "plastic container": "envase plastico",
    "food waste": "residuo comida",
    "banana peel": "cascara banano",
    "styrofoam": "icopor",
    "tissue": "servilleta",
    "cigarette butt": "colilla",
    "trash": "basura",
    "garbage": "basura",
}


def parse_args():
    p = argparse.ArgumentParser(description="Detector de desechos con coordenadas relativas al centro de la cámara")
    p.add_argument("--camera", type=int, default=1, help="Índice de la cámara (0, 1, 2...). Por defecto 1 (cámara externa).")
    p.add_argument("--width", type=int, default=1280, help="Ancho de captura")
    p.add_argument("--height", type=int, default=720, help="Alto de captura")
    p.add_argument("--model", type=str, default="yolov8s-worldv2.pt",
                   help="Pesos del modelo. Por defecto YOLO-World (vocabulario abierto). "
                        "Si das pesos propios (best.pt) se usan sus clases.")
    p.add_argument("--classes", type=str, default=None,
                   help="Clases separadas por coma (solo para YOLO-World). Ej: 'plastic bag,plastic bottle'")
    p.add_argument("--conf", type=float, default=0.15, help="Umbral de confianza (0-1). Bájalo si no detecta suficiente.")
    p.add_argument("--rotate-tta", choices=["off", "fallback", "always"], default="fallback",
                   help="Segunda pasada con la imagen girada 90° para detectar objetos acostados "
                        "(p. ej. botellas en horizontal). fallback = solo si la primera pasada no "
                        "detecta nada (rápido); always = siempre (más lento, más robusto); off = desactivado.")
    p.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para NMS")
    p.add_argument("--imgsz", type=int, default=640, help="Tamaño de inferencia")
    p.add_argument("--device", type=str, default=None, help="'cpu', '0' (GPU CUDA), 'mps' (Mac)... Por defecto automático")
    p.add_argument("--flip", action="store_true", help="Espejar la imagen horizontalmente")
    p.add_argument("--px-per-cm", type=float, default=None,
                   help="Calibración opcional: píxeles por cm en el plano de la caneca para mostrar coordenadas en cm")
    p.add_argument("--no-console", action="store_true", help="No imprimir coordenadas en consola")

    # --- Control del carro ---
    p.add_argument("--port", type=str, default=None,
                   help="Puerto serial del Arduino (ej. COM5 o /dev/ttyUSB0). Sin esto solo se simula en pantalla.")
    p.add_argument("--list-ports", action="store_true", help="Listar puertos seriales y salir")
    p.add_argument("--baud", type=int, default=9600, help="Baudios (9600 coincide con el sketch y el HC-06)")
    p.add_argument("--speed", type=int, default=255, help="Velocidad máxima PWM 0-255 (255 = máxima)")
    p.add_argument("--tol", type=int, default=40,
                   help="Mitad del lado (px) de la caja del origen (cuadro amarillo). El carro para cuando "
                        "la caja del objeto la toca; súbelo para parar antes")
    p.add_argument("--turn-speed", type=int, default=180,
                   help="Velocidad máxima PWM al girar izquierda/derecha (menor que --speed para no pasarse por el retraso de la cámara)")
    p.add_argument("--slow-zone", type=int, default=200,
                   help="Zona en píxeles fuera de la tolerancia donde la velocidad baja linealmente al acercarse al objetivo (0 = desactivado)")
    p.add_argument("--min-speed", type=int, default=140,
                   help="Velocidad mínima PWM en la zona lenta. Si el carro se queda quieto antes de llegar, súbela (los motores no giran con PWM muy bajo)")
    p.add_argument("--tol-exit", type=float, default=1.6,
                   help="Una vez alcanzado el objetivo, el carro solo vuelve a moverse si el objeto sale de tol x este factor (evita correcciones al final)")
    p.add_argument("--invert-x", action="store_true", help="Invertir izquierda/derecha")
    p.add_argument("--invert-y", action="store_true", help="Invertir adelante/atrás")
    return p.parse_args()


def open_camera(index, width, height):
    """Abre la cámara usando el backend adecuado según el sistema operativo."""
    system = platform.system()
    if system == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Linux":
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    elif system == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
    return None


def load_model(args):
    try:
        from ultralytics import YOLO, YOLOWorld
    except ImportError:
        print("Falta 'ultralytics'. Instala con: pip install -r requirements.txt")
        sys.exit(1)

    is_world = "world" in os.path.basename(args.model).lower()
    if is_world:
        model = YOLOWorld(args.model)
        classes = [c.strip() for c in args.classes.split(",")] if args.classes else DEFAULT_CLASSES
        model.set_classes(classes)
        print(f"[INFO] YOLO-World cargado con {len(classes)} clases de desecho.")
    else:
        model = YOLO(args.model)
        print(f"[INFO] Modelo personalizado cargado: {args.model}")
        print(f"[INFO] Clases: {list(model.names.values())}")
    return model


def class_color(cls_id):
    """Color estable por clase (BGR)."""
    rng = np.random.RandomState(cls_id * 7 + 3)
    c = rng.randint(60, 255, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def to_origin(cx, cy, w, h, px_per_cm=None):
    """Convierte píxeles de imagen a coordenadas con origen en el centro.
    X: derecha (+). Y: arriba (+)."""
    x = cx - w / 2.0
    y = (h / 2.0) - cy
    if px_per_cm:
        return x / px_per_cm, y / px_per_cm
    return x, y


def draw_reticle(frame):
    """Dibuja ejes y origen (0,0) en el centro de la cámara."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (0, cy), (w, cy), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(frame, (cx, 0), (cx, h), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "(0,0)", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def put_label(frame, text, x, y, color):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(y, th + 6)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def run_model(model, img, args):
    """Inferencia sobre una imagen. Devuelve (boxes xyxy, confs, clases)."""
    r = model.predict(img, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=args.device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int))
    return (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


def detect(model, frame, args):
    """Detecta objetos. Los modelos suelen fallar con objetos acostados (una botella
    en horizontal), así que opcionalmente se hace otra pasada con la imagen girada
    90° (donde ese objeto queda vertical) y se mezclan los resultados."""
    boxes, confs, clss = run_model(model, frame, args)

    if args.rotate_tta == "always" or (args.rotate_tta == "fallback" and len(boxes) == 0):
        h = frame.shape[0]
        rot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        b2, c2, k2 = run_model(model, rot, args)
        if len(b2) > 0:
            # Volver las cajas de la imagen girada a la original
            back = np.stack([b2[:, 1], h - b2[:, 2], b2[:, 3], h - b2[:, 0]], axis=1)
            boxes = np.concatenate([boxes, back]).astype(np.float32)
            confs = np.concatenate([confs, c2]).astype(np.float32)
            clss = np.concatenate([clss, k2])
            if len(boxes) > 1:  # quitar duplicados entre las dos pasadas
                import torch
                import torchvision
                keep = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(confs),
                                           args.iou).numpy()
                boxes, confs, clss = boxes[keep], confs[keep], clss[keep]
    return boxes, confs, clss


def main():
    args = parse_args()
    if args.list_ports:
        list_serial_ports()
        return
    model = load_model(args)
    names = model.names
    unit = "cm" if args.px_per_cm else "px"

    cap = open_camera(args.camera, args.width, args.height)
    if cap is None:
        print(f"[ERROR] No se pudo abrir la cámara con índice {args.camera}. "
              f"Prueba con --camera 1 o --camera 2.")
        sys.exit(1)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Cámara abierta: {real_w}x{real_h}. Presiona 'q' para salir.")

    car = None
    if args.port:
        try:
            car = CarController(args.port, args.baud)
            print(f"[INFO] Carro conectado en {args.port} @ {args.baud} baudios.")
        except Exception as e:
            print(f"[ERROR] No se pudo abrir el puerto {args.port}: {e}")
            print("Revisa el puerto con --list-ports y cierra el Monitor Serie del IDE de Arduino.")
            cap.release()
            sys.exit(1)
    else:
        print("[INFO] Sin --port: modo simulación (no se envían comandos al carro).")
    paused = False
    reached = False  # ya llegó al objetivo (para la histéresis de la tolerancia)

    os.makedirs("capturas", exist_ok=True)
    print_console = not args.no_console
    prev_time = time.time()
    fps = 0.0
    camera_reading_failed = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if not camera_reading_failed:
                print("[AVISO] No se pudo leer un frame de la cámara; esperando el siguiente...")
                camera_reading_failed = True
            reached = False
            if car is not None:
                car.stop()
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            time.sleep(0.05)
            continue
        if camera_reading_failed:
            print("[INFO] La cámara volvió a entregar frames.")
            camera_reading_failed = False
        if args.flip:
            frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]

        boxes, confs, clss = detect(model, frame, args)

        draw_reticle(frame)

        detections = []
        target_px = None   # (X, Y) en píxeles del centro del objeto elegido
        target_box = None  # (xmin, xmax, ymin, ymax) de su caja, origen en el centro de la cámara
        if len(boxes) > 0:
            # Quedarse solo con el objeto más grande (el más visible)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            boxes, confs, clss = boxes[best:best + 1], confs[best:best + 1], clss[best:best + 1]

            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clss):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                X, Y = to_origin(cx, cy, w, h, args.px_per_cm)
                target_px = to_origin(cx, cy, w, h)
                target_box = (x1 - w / 2.0, x2 - w / 2.0, h / 2.0 - y2, h / 2.0 - y1)

                name = names[cls_id]
                name_es = ES.get(name, name)
                color = class_color(cls_id)

                # Caja, centro y línea desde el origen hasta el objeto
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.line(frame, (w // 2, h // 2), (cx, cy), color, 1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 1, cv2.LINE_AA)

                put_label(frame, f"{name_es} {conf:.0%}", x1, y1, color)
                coord_txt = f"({X:+.1f}, {Y:+.1f}) {unit}"
                cv2.putText(frame, coord_txt, (cx + 10, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, coord_txt, (cx + 10, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

                detections.append((name_es, float(conf), X, Y))

        # ---- Control del carro: llevar el objeto a (0, 0) ----
        cmd, speed, status = "S", 0, "SIN OBJETO"
        if target_box is not None:
            # Para cuando la caja del objeto toca la caja del origen (cuadro amarillo).
            # Histéresis: tras llegar, esa caja se ensancha para no corregir de más.
            tol_eff = args.tol * args.tol_exit if reached else args.tol
            cmd, gap = decide_command_box(target_box, tol_eff, args.invert_x, args.invert_y)
            if cmd == "S":
                reached = True
                status = "OBJETIVO ALCANZADO"
            else:
                reached = False
                # Girar más lento que avanzar/retroceder; todo baja al acercarse
                max_sp = args.turn_speed if cmd in ("L", "R") else args.speed
                speed = speed_for(gap, 0, max_sp, args.min_speed, args.slow_zone)
                status = CMD_NAMES[cmd]
        else:
            reached = False
        if paused:
            cmd, speed, status = "S", 0, "PAUSADO (espacio para reanudar)"
        if car is not None:
            car.set(cmd, speed)

        # Zona de tolerancia alrededor de (0,0) y barra de estado inferior
        cv2.rectangle(frame, (w // 2 - args.tol, h // 2 - args.tol),
                      (w // 2 + args.tol, h // 2 + args.tol), (0, 255, 255), 1)
        cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
        mode_txt = "CARRO" if car is not None else "SIMULACION"
        bar_color = (0, 255, 0) if cmd != "S" else (0, 255, 255)
        cv2.putText(frame, f"{mode_txt} | CMD: {cmd}{speed} | {status}", (8, h - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bar_color, 1, cv2.LINE_AA)

        # Panel de información
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        cv2.rectangle(frame, (0, 0), (330, 28 + 20 * min(len(detections), 8)), (0, 0, 0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Objetos: {len(detections)}", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        for i, (n, c, X, Y) in enumerate(detections[:8]):
            cv2.putText(frame, f"{i + 1}. {n}: ({X:+.1f}, {Y:+.1f}) {unit}", (8, 40 + 20 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        if print_console and detections:
            line = " | ".join(f"{n} ({X:+.1f}, {Y:+.1f}){unit}" for n, _, X, Y in detections)
            print(line)

        cv2.imshow("Detector de desechos - (0,0) = centro de la camara", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("s"):
            path = os.path.join("capturas", f"captura_{datetime.now():%Y%m%d_%H%M%S}.png")
            cv2.imwrite(path, frame)
            print(f"[INFO] Captura guardada: {path}")
        elif key == ord("c"):
            print_console = not print_console
        elif key == ord(" "):
            paused = not paused
            print("[INFO] PAUSADO" if paused else "[INFO] REANUDADO")

    if car is not None:
        car.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
