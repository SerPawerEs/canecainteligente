"""
Control del carro por puerto serial (Arduino + L298N).

Funciona igual por cable USB o por Bluetooth HC-06 (el HC-06 aparece como un
puerto COM en el PC; baudios por defecto 9600).

Protocolo hacia el Arduino: una línea de texto por comando, ej. "F255\\n"
    F = adelante, B = atrás, L = izquierda, R = derecha, S = alto
"""

import atexit
import threading
import time

import serial
from serial.tools import list_ports

CMD_NAMES = {
    "F": "ADELANTE",
    "B": "ATRAS",
    "L": "IZQUIERDA",
    "R": "DERECHA",
    "S": "ALTO",
}


def list_serial_ports():
    """Imprime los puertos seriales disponibles."""
    ports = list(list_ports.comports())
    if not ports:
        print("No se encontraron puertos seriales.")
        return
    print("Puertos seriales disponibles:")
    for p in ports:
        print(f"  {p.device}  -  {p.description}")


def decide_command(x, y, tol=40, invert_x=False, invert_y=False):
    """Decide el movimiento a partir de la posición del objeto.

    x, y: coordenadas del objeto en píxeles con el origen en el centro de la
          cámara (X positivo = derecha, Y positivo = arriba).

    Reglas:
        y > 0 -> F (avanzar)      y < 0 -> B (retroceder)
        x < 0 -> L (izquierda)    x > 0 -> R (derecha)
    (Y estaba invertido con el montaje real de la cámara; con invert_y=True
    se vuelve a la regla original: y < 0 avanzar, y > 0 retroceder.)

    Se corrige primero el eje con mayor error. Devuelve (cmd, error_px);
    cmd = 'S' cuando el objeto ya está dentro de la tolerancia (zona 0,0).
    """
    if invert_x:
        x = -x
    if invert_y:
        y = -y

    out_x = abs(x) > tol
    out_y = abs(y) > tol
    if not out_x and not out_y:
        return "S", 0.0

    if out_x and (not out_y or abs(x) >= abs(y)):
        return ("L" if x < 0 else "R"), abs(x)
    return ("F" if y > 0 else "B"), abs(y)


def decide_command_box(box, tol=40, invert_x=False, invert_y=False):
    """Como decide_command, pero para en cuanto la CAJA del objeto toca la caja
    del origen (el cuadrado de lado 2*tol centrado en (0,0)); cualquier punto de
    la caja del objeto sirve, no hace falta centrarlo.

    box: (xmin, xmax, ymin, ymax) de la caja del objeto en píxeles, con el origen
         en el centro de la cámara (X positivo = derecha, Y positivo = arriba).

    Devuelve (cmd, distancia_px): la distancia es el hueco que falta entre ambas
    cajas en el eje elegido (el de mayor hueco). cmd = 'S' si ya se tocan.
    """
    xmin, xmax, ymin, ymax = box
    if invert_x:
        xmin, xmax = -xmax, -xmin
    if invert_y:
        ymin, ymax = -ymax, -ymin

    # Hueco entre la caja del objeto y la caja del origen (0 si se solapan en ese eje)
    gap_x = max(xmin - tol, -tol - xmax, 0.0)
    gap_y = max(ymin - tol, -tol - ymax, 0.0)
    if gap_x == 0 and gap_y == 0:
        return "S", 0.0

    if gap_x >= gap_y:
        return ("R" if xmin > tol else "L"), gap_x
    return ("F" if ymin > tol else "B"), gap_y


def speed_for(error, tol, max_speed=255, min_speed=150, slow_zone=0):
    """Velocidad PWM (0-255). Por defecto siempre la máxima.

    Si slow_zone > 0, la velocidad baja linealmente hasta min_speed cuando el
    objeto está dentro de (tol + slow_zone) píxeles del centro, para no pasarse.
    """
    if slow_zone <= 0:
        return int(max_speed)
    min_speed = min(min_speed, max_speed)
    frac = min(max((error - tol) / float(slow_zone), 0.0), 1.0)
    return int(min_speed + (max_speed - min_speed) * frac)


class CarController:
    """Envía continuamente el último comando al Arduino en un hilo aparte.

    Así el carro sigue el comando entre frames aunque la inferencia sea lenta.
    Seguridad doble:
      - Si el programa no actualiza el comando en `max_age` segundos, envía ALTO.
      - El Arduino se detiene solo si deja de recibir comandos (TIMEOUT_MS).
    """

    def __init__(self, port, baud=9600, send_hz=20.0, max_age=0.5, boot_wait=2.0):
        self.port = port
        self.error = None
        self._hz = send_hz
        self._max_age = max_age
        self._cmd = "S"
        self._speed = 0
        self._last_set = time.time()
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._closed = False

        self._ser = serial.serial_for_url(port, baudrate=baud, timeout=0.1, write_timeout=0.5)
        # Al abrir el puerto USB el Arduino se reinicia: esperar a que arranque
        time.sleep(boot_wait)
        self._ser.reset_input_buffer()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    # ---------------- API pública ----------------
    def set(self, cmd, speed=255):
        """Fija el comando actual (F, B, L, R, S) y la velocidad 0-255."""
        cmd = cmd.upper()
        if cmd not in CMD_NAMES:
            raise ValueError(f"Comando inválido: {cmd}")
        speed = max(0, min(255, int(speed)))
        with self._lock:
            self._cmd = cmd
            self._speed = 0 if cmd == "S" else speed
            self._last_set = time.time()

    def stop(self):
        self.set("S", 0)

    def run_for(self, cmd, speed, seconds):
        """Mantiene un comando durante `seconds` segundos (para pruebas)."""
        end = time.time() + seconds
        while time.time() < end:
            self.set(cmd, speed)
            time.sleep(0.05)

    def close(self):
        """Detiene el carro y cierra el puerto (se llama solo al salir)."""
        if self._closed:
            return
        self._closed = True
        self._stop_evt.set()
        try:
            self._thread.join(timeout=1.0)
        except RuntimeError:
            pass
        try:
            for _ in range(3):
                self._ser.write(b"S0\n")
                self._ser.flush()
                time.sleep(0.02)
            self._ser.close()
        except Exception:
            pass

    # ---------------- Interno ----------------
    def _loop(self):
        period = 1.0 / self._hz
        while not self._stop_evt.is_set():
            with self._lock:
                cmd, speed = self._cmd, self._speed
                age = time.time() - self._last_set
            if cmd != "S" and age > self._max_age:
                cmd, speed = "S", 0
            try:
                self._ser.write(f"{cmd}{speed}\n".encode("ascii"))
            except serial.SerialException as e:
                self.error = str(e)
                print(f"[ERROR] Se perdió la conexión con el carro: {e}")
                return
            self._stop_evt.wait(period)
