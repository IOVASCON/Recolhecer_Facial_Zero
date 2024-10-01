import dlib
import cv2
import matplotlib.pyplot as plt

# Carregar o detector de face pré-treinado
detector = dlib.get_frontal_face_detector()

# Carregar uma imagem para teste (substitua pelo caminho de uma imagem salva)
image_path = "L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/data/images/train/Maria/face1.jpg"  # Altere o nome conforme necessário
image = cv2.imread(image_path)

# Converter para escala de cinza (o dlib trabalha melhor com imagens em preto e branco)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar faces na imagem
faces = detector(gray_image)

# Desenhar retângulos em torno das faces detectadas
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir a imagem com as faces detectadas
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
