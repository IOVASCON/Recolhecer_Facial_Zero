# Projeto de Reconhecimento Facial com Classificação de Faces

## Objetivo

O objetivo deste projeto é criar um sistema de **reconhecimento facial** que detecta e classifica faces humanas a partir de imagens. Usamos bibliotecas como **OpenCV**, **TensorFlow/Keras**, e **Haar Cascade** para detecção e classificação das faces. O projeto consiste em carregar imagens de rostos humanos, detectar as faces presentes, classificá-las utilizando um modelo de aprendizado de máquina previamente treinado e salvar o resultado, incluindo a imagem com as faces detectadas e o rótulo da classe correspondente.

## Estrutura do Projeto

O projeto está organizado da seguinte maneira:

    L:\VSCode\PYTHON\DIO\Reconhecer_Facial_Zero
    │
    ├── .venv
    │
    ├── data/                             # Contém as imagens de teste e treino
    │   ├── processed/                    # Imagens processadas ou manipuladas
    │   ├── raw/                          # Imagens brutas que serão processadas
    │   └── images
    │       ├── test
    │       │   ├── know
    │       │   ├── unknown
    │       └── train
    │           ├── Carol
    │           └── Danilo
    │           ├── Gustavo
    │           └── Maria
    │           ├── Rodrigo
    │           └── Thalita
    │           ├── unknown
    |
    ├── models/                           # Redes neurais treinadas para detecção e classificação
    │   ├── face_detection_model.h5       # Modelo treinado para detecção de faces
    │   └── face_classification_model.h5  # Modelo treinado para reconhecimento facial
    │
    ├── notebooks/                        # Notebooks Jupyter com experimentos e visualizações
    │   └── face_detection.ipynb          # Notebooks com código para análise de faces
    │
    |__ prints_projeto                    # prints do projeto
    |
    ├── src/                              # Código-fonte do projeto
    │   ├── __init__.py                   # Inicializador do pacote
    │   ├── detect_faces.py               # Script para detectar faces
    │   ├── classify_faces.py             # Script para classificar as faces detectadas
    │   ├── utils.py                      # Funções auxiliares
    │   └── config.py                     # Arquivo de configuração do projeto
    │   └── criar_arquivo.py              # teste de criação de arquivo
    │
    ├── tests/                            # Testes unitários e funcionais
    │   ├── test_detect_faces.py          # Testes para a detecção de faces
    │   └── test_classify_faces.py        # Testes para a classificação de faces
    │   └── test_model_access.py          # Teste criação modelo
    │   └── test_simples_bibliotecas_instaladas.py        # teste de verificação de bibliotecas criadas
    │
    ├── requirements.txt                  # Bibliotecas necessárias para rodar o projeto
    ├── .gitignore                        # arquivos ignorados ao GitHub
    └── README.md                         # Documentação e instruções do projeto

## Descrição dos arquivos principais

1. data/images/train: Essa pasta contém as subpastas person1, person2, person3 e unknown, onde você vai salvar as imagens usadas para treinar o modelo.

- person1, person2, person3: Aqui você coloca múltiplas imagens de cada pessoa, usadas para treinamento.
- unknown: Imagens de rostos que o modelo não conhece, para que ele possa aprender a identificar rostos desconhecidos.

2. data/images/test: Essas pastas contêm as imagens usadas para testar o modelo após o treinamento.

- known: Imagens de pessoas que já foram usadas no treinamento, para validar o reconhecimento.
- unknown: Imagens de pessoas que o modelo não conhece.

    data/raw/: Contém as imagens originais que serão usadas no projeto, como as de treinamento e validação.

    models/: Guardará os modelos já treinados, tanto para detecção de faces quanto para a classificação.

    notebooks/: Usaremos essa pasta para criar Jupyter Notebooks onde serão feitos os testes preliminares, visualização dos resultados e ajustes.

    prints_projeto/: Contém prints diversos do projeto

    src/detect_faces.py: Script Python responsável por carregar o modelo treinado de detecção e aplicar a detecção em uma imagem ou conjunto de imagens.

    src/classify_faces.py: Este script será responsável por classificar as faces detectadas utilizando a rede de classificação.

    src/utils.py: Funções auxiliares, como pré-processamento de imagens, manipulação de arquivos, etc.

    src/config.py: Arquivo de configuração, onde definimos os caminhos para os arquivos, parâmetros de rede e configurações gerais do projeto.

    tests/: Contém os testes unitários para verificar a detecção e classificação das faces.

    requirements.txt: Lista de bibliotecas necessárias para rodar o projeto. Exemplo:

A pasta "images" foi criada para organizar de forma clara e sistemática as imagens utilizadas no treinamento e teste do modelo de reconhecimento facial. Aqui estão as razões por trás dessa estrutura:

    Organização entre dados brutos e processados:
        raw/: Guarda as imagens brutas, ou seja, aquelas que ainda não foram processadas ou manipuladas.
        processed/: Guarda as imagens que já passaram por alguma manipulação ou transformação (como redimensionamento, aumento de contraste, etc.).

    Separação entre dados de treino e teste:
        train/: Contém as imagens que o modelo usará para aprender (ou treinar). Aqui, as imagens são organizadas por classes (exemplo: person1, person2, unknown).
        test/: Contém as imagens usadas para avaliar o desempenho do modelo, separadas entre rostos conhecidos (known) e desconhecidos (unknown).

    Facilita a expansão do projeto:
        Se no futuro você precisar adicionar mais classes (mais pessoas, por exemplo), a organização em pastas como person1, person2, etc., torna mais fácil fazer isso.
        A pasta unknown permite adicionar imagens de rostos desconhecidos, uma classe importante em problemas de reconhecimento facial.

Em resumo, essa estrutura garante uma separação clara entre os dados usados para diferentes etapas do treinamento e teste do modelo.

## Estrutura Pastas Testes

    data/
    └── images/
        ├── test/
        │   ├── Maria/       # Imagens que você já possui da Maria
        │   ├── person1/     # Adicione imagens de outra pessoa aqui
        │   ├── person2/     # Adicione imagens de mais uma pessoa aqui
        │   └── unknown/     # Imagens de rostos desconhecidos
        └── train/
            ├── person1/       # Imagens de Maria ou de outras pessoas conhecidas
            └── unknown/     # Imagens de rostos desconhecidos que não pertencem a nenhuma das pessoas no conjunto de treino

### Orientações

- **Pastas "person1" e "person2":**
        Adicione imagens de outras pessoas nessas pastas. Pode ser qualquer outra pessoa que você queira que o modelo identifique.
        Se você não tiver imagens agora, pode duplicar as imagens da Maria em uma dessas pastas temporariamente, só para fins de teste.

- **Pasta "unknown":**
        Coloque imagens de rostos que o modelo deve classificar como desconhecidos. Essas podem ser de qualquer pessoa que não se encaixe em "Maria", "person1", ou "person2".

- **Testar a detecção de rostos:**
        Depois de organizar as pastas, você pode testar a detecção com o script detect_faces.py. Ele deve processar as imagens e tentar detectar os rostos de acordo com os dados fornecidos.

## Tecnologias Utilizadas

- **Python 3.10** com ambiente virtual (venv)
- **TensorFlow/Keras** para criação e treinamento do modelo de classificação facial.
- **OpenCV** para a detecção de faces usando Haar Cascades.
- **Conda** para instalação de pacotes específicos.
- **H5Py** para leitura/escrita do modelo de classificação salvo no formato `.h5`.

## Instalação e Dependências

- **tensorflow==2.17.0** Versão do TensorFlow para o modelo de redes neurais
- **numpy==1.26.4** Para operações matemáticas e manipulação de arrays
- **opencv-python==4.5.4.60** Para manipulação de imagens e vídeos
- **matplotlib==3.5.1** Para visualização de imagens e gráficos
- **Pillow==9.0.0** Manipulação de imagens no formato PIL
- **scipy==1.7.3** Biblioteca científica para suporte a funções e otimizações
- **dlib==19.22.0** Biblioteca opcional para detecção facial com modelos pré-treinados
- **h5py==3.6.0** Para trabalhar com arquivos de modelo HDF5

## Ambiente Virtual

Trabalhar com ambientes virtuais é uma prática recomendada para projetos Python, pois permite isolar as dependências de um projeto de outros que possam estar instalados no sistema. Isso ajuda a evitar conflitos de versões entre bibliotecas.

1. Criar o Ambiente Virtual

No terminal, navegue até o diretório do seu projeto:
cd L:\VSCode\PYTHON\DIO\Reconhecer_Facial_Zero

executar o comando para criar o ambiente virtual:
python -m venv .venv

Isso criará um ambiente virtual na pasta .venv dentro do diretório do seu projeto.

2. Ativar o Ambiente Virtual - Windows

Dependendo do seu sistema operacional, a ativação do ambiente virtual é diferente:
.\.venv\Scripts\activate

Uma vez ativado, uma pasta algo como (.venv) será exibida no início da linha de comando, indicando que o ambiente virtual está ativo.

3. Instalar Dependências no Ambiente Virtual

Com o ambiente ativado, instale as dependências necessárias para o projeto. Isso pode ser feito usando o arquivo requirements.txt que já criamos anteriormente:
pip install -r requirements.txt

Isso instalará todas as bibliotecas listadas no arquivo requirements.txt dentro do ambiente virtual.

4. Verificar as Instalações

Para garantir que todas as bibliotecas foram instaladas corretamente, verificar as dependências instaladas com o comando:
python src/detect_faces.py

6. Desativar o Ambiente Virtual

Quando terminar de trabalhar, você pode desativar o ambiente virtual com o comando:
deactivate

7. Adicionar .venv ao .gitignore

Se você estiver usando Git para controle de versão, é uma boa prática adicionar a pasta .venv ao seu arquivo .gitignore para que o ambiente virtual não seja incluído no repositório. No arquivo .gitignore, adicione:
.venv/

## Instalação e Dependências

1. Clone o repositório:
git clone https://

2.  Navegue para o diretório do projeto e ative o ambiente virtual:
cd Reconhecer_Facial_Zero source .venv/bin/activate (ou .venv\Scripts\activate no Windows)

3. Instale as dependências listadas em `requirements.txt`:
pip install -r requirements.txt

4. Certifique-se de que as bibliotecas estão corretamente instaladas rodando os testes:
python tests/test_simples_bibliotecas_instaladas.py

## Execução

### Detecção de Faces

Para detectar faces em imagens, execute o seguinte comando:
python src/detect_faces.py

O script irá processar as imagens no diretório `data/images/train` e salvar o resultado no diretório `processed`.

### Classificação de Faces

Para classificar faces usando o modelo pré-treinado, execute:
python src/classify_faces.py

Este script detectará as faces nas imagens, classificará as faces e salvará a imagem com as classificações no diretório `data/processed`.

## Observações

- As imagens devem estar organizadas em subdiretórios dentro de `data/images/train` para que o sistema possa processá-las adequadamente.
- Certifique-se de que o modelo de classificação está devidamente treinado e salvo no caminho correto (`models/face_classification_model.h5`).
- O arquivo `criar_arquivo.py` foi utilizado para gerar e salvar o modelo de classificação de faces. Caso necessário, ele pode ser executado para recriar o modelo.

## Licença

Este projeto está licenciado sob os termos da licença MIT. Sinta-se à vontade para usar, modificar e compartilhar conforme necessário.
