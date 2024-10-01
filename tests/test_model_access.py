import h5py

# Caminho absoluto para o arquivo .h5
model_path = "L:/VSCode/PYTHON/DIO/Reconhecer_Facial_Zero/models/face_classification_model.h5"

# Tentar abrir o arquivo
try:
    with h5py.File(model_path, 'r') as f:
        print("Arquivo .h5 acessado com sucesso!")
except OSError as e:
    print(f"Erro ao acessar o arquivo: {e}")
