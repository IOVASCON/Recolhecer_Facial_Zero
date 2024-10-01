import cv2
import os

# Carregar o modelo Haar Cascade para detecção de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_path, output_directory):
    """Função para detectar faces em uma imagem e salvar o resultado"""
    # Carregar a imagem
    img = cv2.imread(image_path)
    
    # Verificar se a imagem foi carregada corretamente
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return 0  # Retorna 0 faces detectadas se a imagem não for carregada
    
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detectar faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Desenhar retângulos em torno das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Salvar a imagem com as faces detectadas
    output_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    # Retorna o número de faces detectadas
    return len(faces)

def process_directory(directory_path, output_directory):
    """Função para processar todas as imagens em um diretório e contar o total de faces detectadas"""
    total_faces = 0
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            print(f"Processando imagem: {image_path}")
            # Detectar faces e salvar a imagem com o resultado
            faces_detected = detect_faces(image_path, output_directory)
            total_faces += faces_detected
            print(f"Faces detectadas na imagem {filename}: {faces_detected}")
    
    # Exibir o total de faces detectadas
    print(f"Total de faces detectadas em todas as imagens: {total_faces}")

# Caminho para o diretório que contém as imagens
directory_path = "L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/data/images/train/Maria"

# Caminho para salvar as imagens processadas
output_directory = "L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/data/processed"

# Garantir que o diretório de saída exista
os.makedirs(output_directory, exist_ok=True)

# Processar todas as imagens no diretório e exibir o resumo final
process_directory(directory_path, output_directory)
