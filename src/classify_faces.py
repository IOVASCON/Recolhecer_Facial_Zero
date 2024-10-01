import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
import os

# Verificação e carregamento do modelo
model_path = 'L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/models/face_classification_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    
# Carregar o modelo de reconhecimento facial (pré-treinado)
model = load_model(model_path)

# Carregar o Haar Cascade para detecção de face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def classify_face(face):
    """Função para classificar uma face detectada"""
    # Pré-processamento da face (redimensionar, normalizar, etc.)
    face = cv2.resize(face, (160, 160))  # Ajuste de tamanho conforme o modelo
    face = face.astype('float32') / 255.0  # Normalizar a imagem
    face = np.expand_dims(face, axis=0)  # Expande as dimensões para compatibilidade com o modelo

    # Fazer a previsão
    prediction = model.predict(face)

    # Retorna a classe e a confiança
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]

    return class_idx, confidence

def recognize_faces(image_path):
    """Função para detectar e reconhecer faces em uma imagem"""
    # Verificar se a imagem existe
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    
    # Carregar a imagem
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Converter a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar faces usando o Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Contador para número de faces detectadas
    face_count = 0

    # Classificar cada face detectada
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        class_idx, confidence = classify_face(face)

        # Exibir o rótulo na imagem com a confiança da previsão
        label = f'Pessoa: {class_idx} ({confidence*100:.2f}%)'
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Contador de faces detectadas
        face_count += 1

    # Exibir o resultado final
    print(f"Total de faces detectadas: {face_count}")
    
    # Salvar a imagem com as classificações e retângulos ao redor das faces
    output_image_path = 'L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/data/processed/faces_recognized.jpg'
    cv2.imwrite(output_image_path, img)
    print(f"Imagem salva em: {output_image_path}")

    # Mostrar a imagem com as faces reconhecidas
    cv2.imshow("Faces reconhecidas", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Testar com uma imagem de exemplo
recognize_faces('L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/data/images/train/Maria/face1.jpg')
