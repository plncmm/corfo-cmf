Clasificador de reclamos - Etapa 1 Reto Corfo CMF
==============================

Organización del repositorio.
------------

    ├── README.md                           <- Documentación del código.
    ├── data                                <- Carpeta con los datos necesarios para ejecutar el proyecto.
    │   ├── cleaned                         <- Dataset resultante luego de procesar los datos originales.
    │   ├── embeddings                      <- Word embeddings utilizados en nuestros experimentos.
    │   ├── input_files                     <- Directorio con los archivos utilizados para el entrenamiento de cada clasificador. 
    │   ├── raw                             <- Archivos originales entregados por la contraparte del proyecto.
    │   └── cleaned                         <- Dataset resultante luego de procesar los datos originales, está desbalanceado.
    │
    ├── logs                                <- Carpeta que contiene los logs del entrenamiento de cada clasificador.
    │
    ├── models                              <- Carpeta que almacena los modelos pre-entrenados.
    │
    ├── src                                 <- Código principal del proyecto.
    │   │
    │   ├── beto_training                   <- Scripts útiles para el entrenamiento y validación del modelo beto.
    │   │   ├── evaluate_beto.py
    │   │   └── train_beto.py
    │   │
    │   ├── datasets                        <- Modulo útil para generar el Dataframe de reclamos, con el procesamiento especificado en el archivo params.yaml
    │   │   └── datasets.py
    │   │
    │   ├── scripts                         <- Scripts de ejecución de cada uno de los modelos probados para la clasificación.
    │   │   ├── run_beto.py
    │   │   ├── run_cnn.py
    │   │   ├── run_logistic.py
    │   │   ├── run_lstm.py
    │   │   ├── run_nv.py
    │   │   ├── run_random_forest.py
    │   │   ├── run_roberta.py
    │   │   ├── run_svm.py
    │   │   ├── run_xgboost.py
    │   │   └── ner_matcher.py              <- Clasificador en proceso de desarrollo para el caso de nombre de entidades.
    │   │
    │   ├── utils                           <- Scripts útiles para el entrenamiento de los modelos, y el formato de datos.
    │   │   ├── beto_cross_validation.py 
    │   │   ├── beto_utils.py 
    │   │   ├── create_input.py 
    │   │   ├── format_data.py
    │   │   ├── general_utils.py
    │   │   ├── lstm_cnn_utils.py
    │   │   └── sklearn_utils.py
    │   │
    │   ├── main.py                         <- Script que ejecuta el modelo según lo específicado en el archivo yaml.
    │   └── beto_infer.py                   <- Script utilizado para predecir las etiquetas de los datos sin clasificación.
    │ 
    ├── params.yaml                         <- Archivo que contiene todas las configuraciones del proyecto.
    │
    └── requirements.txt                    <- Archivo con los requerimientos de librerías para poder ejecutar el proyecto.

Ejecución.
------------

1. Luego de haber clonado el repositorio, crear un ambiente con el comando: `python -m venv venv` y activarlo.

2. Instalar los requerimientos, para ello basta con que ejecuten el comando `pip install -r requirements.txt` Comprobar que los siguientes paquetes estén instalados: Keras, Tensorflow, Pandas, Spacy, Pyyaml, Nltk, Matplotlib, Seaborn, Sadice, Sklearn, Openpyxl, Imblearn, Gensim.

3. Descargar el modelo estadístico utilizado para tokenizar: `pip install spacy` y `python -m spacy download es_core_news_sm`

4. Si quieren utilizar una gpu NVIDIA, pueden utilizar el siguiente comando para instalar la librería pytorch: `pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`, esto para el caso del modelo BETO, que requiere harta potencia.

5. Crear una carpeta data, siguiendo la estructura presentada anteriormente. Situar tanto los archivos en formato excel como también los embeddings. En cuanto a los archivos, se partió de la base de los 2 archivos entregados por la contraparte de la cmf y se procesaron con el Script `format_data.py`, con esto es posible generar el archivo para la carpeta cleaned. El resto de archivos lo pueden solicitar al correo matirojasga@gmail.com. Con respecto a los embeddings, se utilizaron los siguientes: https://github.com/crscardellino/sbwce, en la versión Word2Vec.

6. Modificar los parámetros que deseen en el archivo params.yaml. Importante: El modelo a utilizar se elige al comienzo en el parámetro model, y más adelante salen los parámetros específicos de cada modelo.

