#!/usr/bin/env python3
"""
Prueba de motores por cable (sin cámara).

IMPORTANTE: levanta las ruedas del suelo antes de empezar.

Uso:
    python test_car.py --list-ports
    python test_car.py --port COM5
    python test_car.py --port /dev/ttyUSB0 --speed 255 --seconds 1.5

Hace la secuencia ADELANTE, ATRAS, IZQUIERDA, DERECHA. Cada paso dura unos
segundos con una pausa entre pasos.
"""

import argparse

from car_controller import CarController, list_serial_ports

STEPS = [
    ("F", "ADELANTE", "las dos ruedas giran hacia adelante"),
    ("B", "ATRAS", "las dos ruedas giran hacia atrás"),
    ("L", "IZQUIERDA", "rueda izquierda hacia atrás y derecha hacia adelante"),
    ("R", "DERECHA", "rueda derecha hacia atrás e izquierda hacia adelante"),
]

HELP = """
Si algo salió al revés, ajusta las constantes al inicio de car_control.ino y vuelve a subirlo:
  - ADELANTE y ATRAS salieron invertidos en las dos ruedas -> INVERT_LEFT = true e INVERT_RIGHT = true
  - Solo una rueda gira al revés en ADELANTE              -> INVERT_LEFT o INVERT_RIGHT (la de esa rueda)
  - ADELANTE/ATRAS bien, pero IZQUIERDA y DERECHA al revés -> SWAP_SIDES = true
"""


def main():
    p = argparse.ArgumentParser(description="Prueba de motores del carro")
    p.add_argument("--port", type=str, default=None, help="Puerto serial del Arduino (ej. COM5)")
    p.add_argument("--list-ports", action="store_true", help="Listar puertos seriales y salir")
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--speed", type=int, default=255, help="PWM 0-255")
    p.add_argument("--seconds", type=float, default=1.5, help="Duración de cada paso")
    args = p.parse_args()

    if args.list_ports:
        list_serial_ports()
        return
    if not args.port:
        p.error("Falta --port (usa --list-ports para ver los disponibles)")

    car = CarController(args.port, args.baud)
    try:
        input("Levanta las ruedas del suelo y presiona Enter para empezar...")
        for cmd, name, expected in STEPS:
            print(f"-> {name}: deberían verse {expected}")
            car.run_for(cmd, args.speed, args.seconds)
            car.run_for("S", 0, 1.0)
        print("Prueba terminada.")
        print(HELP)
    except KeyboardInterrupt:
        print("\nInterrumpido.")
    finally:
        car.close()


if __name__ == "__main__":
    main()
