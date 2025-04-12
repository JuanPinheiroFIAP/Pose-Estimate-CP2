const int pin1 = 7;  // Vermelho - Mão Direita
const int pin2 = 6;  // Verde - Mão Esquerda
const int pin3 = 5;  // Azul - Olho Direito
const int pin4 = 4;  // Amarelo - Olho Esquerdo

void setup() {
  // Inicializa a comunicação serial a 9600 bps
  Serial.begin(9600);
  
  // Configura os pinos 4, 6, 5 e 7 como saída
  pinMode(pin1, OUTPUT);
  pinMode(pin2, OUTPUT);
  pinMode(pin3, OUTPUT);
  pinMode(pin4, OUTPUT);
  
  // Desliga todos os LEDs inicialmente
  digitalWrite(pin1, LOW);
  digitalWrite(pin2, LOW);
  digitalWrite(pin3, LOW);
  digitalWrite(pin4, LOW);
}

void loop() {
  if (Serial.available() > 0) {
    // Lê o caractere enviado via Serial
    char command = Serial.read();
    
    // Converte o caractere para inteiro, se necessário
    int commandInt = command - '0';  // Converte o caractere para o valor inteiro

    // Verifica o comando recebido e liga o pino correspondente
    if (commandInt == 1) {
      digitalWrite(pin1, HIGH);  // Mão direita
      digitalWrite(pin2, LOW);
      digitalWrite(pin3, LOW);
      digitalWrite(pin4, LOW);
      Serial.println("Mão direita levantada");
    } else if (commandInt == 2) {
      digitalWrite(pin2, HIGH);  // Mão esquerda
      digitalWrite(pin1, LOW);
      digitalWrite(pin3, LOW);
      digitalWrite(pin4, LOW);
      Serial.println("Mão esquerda levantada");
    } else if (commandInt == 3) {
      digitalWrite(pin3, HIGH);  // Olho direito
      digitalWrite(pin1, LOW);
      digitalWrite(pin2, LOW);
      digitalWrite(pin4, LOW);
      Serial.println("Olho direito fechado");
    } else if (commandInt == 4) {
      digitalWrite(pin4, HIGH);  // Olho esquerdo
      digitalWrite(pin1, LOW);
      digitalWrite(pin2, LOW);
      digitalWrite(pin3, LOW);
      Serial.println("Olho esquerdo fechado");
    } else{
      // Desliga todos os LEDs inicialmente
      digitalWrite(pin1, LOW);
      digitalWrite(pin2, LOW);
      digitalWrite(pin3, LOW);
      digitalWrite(pin4, LOW);
    }
  }
}
