/*
 * Control de carro con L298N por comandos seriales (cable USB o HC-06).
 *
 * Protocolo (texto, cada comando termina en salto de linea):
 *   F<vel>  adelante
 *   B<vel>  atras
 *   L<vel>  giro a la izquierda (sobre su eje)
 *   R<vel>  giro a la derecha   (sobre su eje)
 *   S       alto (freno)
 *   <vel> = 0..255. Si se omite se usa 255 (maxima velocidad).
 *   Ejemplo: "F255\n"
 *
 * Seguridad: si pasan mas de TIMEOUT_MS sin recibir ningun comando,
 * el carro se detiene solo (por si se cierra el programa o se corta el cable).
 *
 * IMPORTANTE: los pines 0 y 1 son el Serial de hardware y se comparten con el USB.
 * Desconecta los cables del HC-06 (TXD/RXD) de los pines 0 y 1 mientras
 * subes el sketch o trabajas por cable USB.
 *
 * Nota: el Arduino IDE pide que el .ino este dentro de una carpeta con su
 * mismo nombre ("car_control"). Si te lo pregunta, acepta que la cree.
 */

// ---------- Pines (segun tu conexion) ----------
const uint8_t IN1 = 8;   // Motor 1
const uint8_t IN2 = 7;   // Motor 1
const uint8_t IN3 = 6;   // Motor 2
const uint8_t IN4 = 5;   // Motor 2
const uint8_t ENA = 11;  // PWM Motor 1
const uint8_t ENB = 3;   // PWM Motor 2

// ---------- Ajustes ----------
const long BAUD = 9600;                    // 9600 = velocidad por defecto del HC-06
const unsigned long TIMEOUT_MS = 500;      // parada automatica sin comandos
const bool BRAKE_ON_STOP = true;           // true = freno activo, false = rueda libre

// Motor 1 = lado IZQUIERDO y Motor 2 = lado DERECHO.
// Si al probar giran al reves, pon SWAP_SIDES en true.
const bool SWAP_SIDES = false;
// Si al ir "adelante" una rueda gira hacia atras, invierte ese lado.
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = false;

// ---------- Estado ----------
unsigned long lastCmdMs = 0;
bool moving = false;

char buf[12];
uint8_t len = 0;

void setMotor(uint8_t inA, uint8_t inB, uint8_t en, int speed) {
  speed = constrain(speed, -255, 255);
  if (speed > 0) {
    digitalWrite(inA, HIGH);
    digitalWrite(inB, LOW);
    analogWrite(en, speed);
  } else if (speed < 0) {
    digitalWrite(inA, LOW);
    digitalWrite(inB, HIGH);
    analogWrite(en, -speed);
  } else if (BRAKE_ON_STOP) {
    digitalWrite(inA, LOW);   // ambas entradas iguales + EN alto = freno
    digitalWrite(inB, LOW);
    analogWrite(en, 255);
  } else {
    digitalWrite(inA, LOW);   // EN bajo = rueda libre
    digitalWrite(inB, LOW);
    analogWrite(en, 0);
  }
}

// left / right: -255..255 (positivo = adelante)
void drive(int left, int right) {
  if (INVERT_LEFT) left = -left;
  if (INVERT_RIGHT) right = -right;
  int m1 = SWAP_SIDES ? right : left;
  int m2 = SWAP_SIDES ? left : right;
  setMotor(IN1, IN2, ENA, m1);
  setMotor(IN3, IN4, ENB, m2);
  moving = (left != 0 || right != 0);
}

void handleCommand(const char *s) {
  char cmd = toupper(s[0]);
  int spd = (s[1] != '\0') ? atoi(s + 1) : 255;
  spd = constrain(spd, 0, 255);

  switch (cmd) {
    case 'F': drive( spd,  spd); break;
    case 'B': drive(-spd, -spd); break;
    case 'L': drive(-spd,  spd); break;
    case 'R': drive( spd, -spd); break;
    case 'S': drive(0, 0);       break;
    default:  return;  // comando desconocido: se ignora
  }
  lastCmdMs = millis();
}

void readSerial() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        buf[len] = '\0';
        handleCommand(buf);
        len = 0;
      }
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      len = 0;  // linea demasiado larga: se descarta
    }
  }
}

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  drive(0, 0);
  Serial.begin(BAUD);
  lastCmdMs = millis();
}

void loop() {
  readSerial();
  if (moving && (millis() - lastCmdMs > TIMEOUT_MS)) {
    drive(0, 0);
  }
}
