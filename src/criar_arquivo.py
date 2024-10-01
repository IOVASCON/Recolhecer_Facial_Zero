import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

# Definir a arquitetura do modelo de classificação de face
def create_face_classification_model():
    model = models.Sequential()
    
    # Camadas convolucionais
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    # Camadas densas
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # Alterar para o número de classes desejado
    
    # Compilando o modelo
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Criar e salvar o modelo
model = create_face_classification_model()
model.save('L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/models/face_classification_model.h5')
print("Modelo salvo com sucesso!")
