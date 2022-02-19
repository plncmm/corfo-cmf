Clasificador de reclamos - Etapa 1 Reto Corfo CMF
==============================

Organización del repositorio.
------------

    ├── README.md          <- Documentación del código.
    ├── data
    │   ├── raw            <- Archivos originales entregados por la contraparte del proyecto.
    │   ├── cleaned        <- Dataset resultante luego de procesar los datos originales, está desbalanceado.
    │   ├── balanced       <- Versión del dataset con clases balanceadas.
    │   ├── min-labels     <- Versión del dataset desbalanceado pero que exige una cantidad mínima de labels.
    │   ├── embeddings     <- Word embeddings utilizados en nuestros experimentos.
    │
    │── logs               <- Carpeta que contiene los logs de cada experimento.
    │
    ├── params.yaml        <- Archivo que contiene todas las configuraciones del proyecto.
    │
    │── models             <- Carpeta que almacena los modelos pre-entrenados.
    │
    ├── reports            
    │   └── report.ppt     <- Presentación con resumen de resultados.
    │
    ├── requirements.txt   <- Archivo con los requerimientos de librerías para poder ejecutar el proyecto.
    │
    ├── src                <- Principal código del proyecto.
    │   │
    │   ├── datasets       <- Modulo útil para generar el Dataframe de reclamos, con el procesamiento requerido.
    │   │   └── datasets.py
    │   │
    │   ├── beto_training  <- Scripts útiles para el entrenamiento e inferencia del modelo beto
    │   │   └── evaluate_beto.py
    │   │   └── infer_beto.py
    │   │   └── train_beto.py
    │   │
    │   ├── models         <- Scripts para entrenar los diferentes modelos de nuestros experimentos.
    │   │   └── run_beto.py
    │   │   └── run_cnn.py
    │   │   └── run_logistic.py
    │   │   └── run_lstm.py
    │   │   └── run_nv.py
    │   │   └── run_roberta.py
    │   │   └── run_svm.py
    │   │   └── run_xgboost.py
    │   │   └── run_xlnet.py
    │   │
    │   │── utils         <- Script útiles para el entrenamiento y validación de los modelos.
    │   │   └── beto_utils.py 
    │   │   └── format_data.py
    │   │   └── general_utils.py
    │   │   └── lstm_cnn_utils.py
    │   │   └── sklearn_utils.py
    │   │
    │   └── main.py        <- Script que ejecuta el modelo según lo específicado en el archivo yaml.

Ejecución.
------------

1. Luego de haber clonado el repositorio, crear un ambiente con el comando: `python -m venv venv` y activarlo.

2. Instalar los requerimientos, para ello basta con que ejecuten el comando `pip install -r requirements.txt`.

3. Descargar el modelo estadístico utilizado para tokenizar: `python -m spacy download es_core_news_sm`

4. Si quieren utilizar una gpu NVIDIA, pueden utilizar el siguiente comando para instalar la librería pytorch: `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html`, esto para el caso del modelo BETO, que requiere harta potencia.

5. Crear una carpeta data, siguiendo la estructura presentada anteriormente. Situar tanto los archivos en formato excel como también los embeddings. En cuanto a los archivos, se partió de la base de los 2 archivos entregados por la contraparte de la cmf y se procesaron con el Script `format_data.py`, con esto es posible generar el archivo para la carpeta cleaned. El resto de archivos lo pueden solicitar al correo matirojasga@gmail.com. Con respecto a los embeddings, se utilizaron los siguientes: https://github.com/crscardellino/sbwce, en la versión Word2Vec.

6. Modificar los parámetros que deseen en el archivos params.yaml. Importante: El modelo a utilizar se elige al comienzo en el parámetro model, y más adelante salen los parámetros específicos de cada modelo.
