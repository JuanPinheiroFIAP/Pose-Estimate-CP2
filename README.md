# 📌 Projeto: Pose Estimation com MediaPipe e Arduino

## 👥 Integrantes do Grupo

| Nome                          | RM      |
|-------------------------------|---------|
| Kaiky Alvaro Miranda          | 98118   |
| Lucas Rodrigues da Silva      | 98344   |
| Juan Pinheiro de França       | 552202  |
| Matheus Gusmão Aragão         | 550826  |
| Júlia Marques Mendes das Neves| 98680   |

---

## 🧠 Objetivo do Projeto

Este projeto tem como objetivo utilizar a biblioteca [MediaPipe](https://developers.google.com/mediapipe) da Google para detectar **movimentos corporais e expressões faciais** em tempo real via webcam. Com isso, enviamos **comandos via Serial para o Arduino Uno**, permitindo que ações físicas sejam controladas a partir da detecção de gestos e piscadas de olhos, por exemplo.

A comunicação com o Arduino permite que os movimentos detectados sejam utilizados para controlar **LEDs, motores ou outros dispositivos eletrônicos** conectados à placa.

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

### 🐍 Python

- Versão: `3.12.9`
- Principais bibliotecas:
  - `opencv-python` (cv2)
  - `mediapipe`
  - `numpy`
  - `pyserial`
  - `time`

> O código Python foi desenvolvido e testado utilizando o **Visual Studio Code (VSCode)**.

### ⚡ Arduino

- **Placa utilizada**: Arduino Uno
- **IDE utilizada**: Arduino IDE
- Comunicação via **Serial (porta COM5 no nosso caso)** com taxa de baud `9600`.

---

## 🔌 Circuito no Tinkercad

- Link para visualização do circuito no Tinkercad:  
🔗 [Tinkercad](https://www.tinkercad.com/things/b2ceVLqVYyk-pose-estimation)

### 🖼️ Visão Esquemática

![Esquemático do Projeto](https://github.com/user-attachments/assets/8120ff84-39de-4cfc-86bc-315426d0b159)

### 🧾 Lista de Componentes

![Componentes do Projeto](https://github.com/user-attachments/assets/203487dc-e311-43ef-967c-716b5a5783e9)

---

## 👀 Visão Geral do Projeto

![Pose Estimation](https://github.com/user-attachments/assets/6436f031-496a-4c99-bb9a-e22f235e665e)

---

## 📄 Estrutura do Projeto

### `app.py` - Código principal com Arduino conectado

Este script realiza:

- Abertura da câmera;
- Detecção de **pose corporal** e **face** com MediaPipe;
- Verificação se a **mão direita ou esquerda está levantada**;
- Verificação se os **olhos estão fechados**;
- Envio de comandos via Serial para o Arduino:
  - `'1'` → Mão direita levantada
  - `'2'` → Mão esquerda levantada
  - `'3'` → Olho direito fechado
  - `'4'` → Olho esquerdo fechado

#### Trecho de exemplo:
```python
if landmarks[16].y < landmarks[12].y:
    print("Mão direita levantada")
    arduino.write(str(1).encode())  # Envia '1' como byte
```

---

### `main.py` - Teste local sem conexão com Arduino

Este script é uma versão alternativa que **não se conecta ao Arduino**. É ideal para testes de reconhecimento visual sem necessidade da placa física.

O comportamento visual e as mensagens no terminal são os mesmos, mas os comandos **não são enviados via Serial**.

---

## 📦 Instalação e Execução

### 1. Clone o repositório
```bash
git clone https://github.com/JuanPinheiroFIAP/Pose-Estimation.git
```

### 2. Instale as dependências do Python
```bash
pip install opencv-python mediapipe pyserial numpy
```

### 3. Execute o script
Para o teste com Arduino:
```bash
python app.py
```

Para o teste local sem Arduino:
```bash
python main.py
```

> **Atenção**: Certifique-se de que a porta Serial correta (`COM5`, por exemplo) está configurada no seu script.

---

## 📋 Comandos Enviados

| Ação Reconhecida         | Código Enviado via Serial |
|--------------------------|----------------------------|
| Mão direita levantada    | `'1'`                      |
| Mão esquerda levantada   | `'2'`                      |
| Olho direito fechado     | `'3'`                      |
| Olho esquerdo fechado    | `'4'`                      |
