Clasificador de reclamos - Etapa 1 Reto Corfo CMF
==============================

Organización del repositorio.
------------
    │
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
    │   │   ├── evaluate_beto.py            <- Script utilizado para la validación y testeo del modelo.
    │   │   └── train_beto.py               <- Script utilizado para el entrenamiento del modelo.  
    │   │
    │   ├── datasets                        <- Modulo útil para generar el Dataframe de reclamos.
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
    ├── README.md                           <- Documentación del código.
    │
    └── requirements.txt                    <- Archivo con los requerimientos de librerías para poder ejecutar el proyecto.

Configuraciones previas.
------------

1. El primer paso es clonar el proyecto. Si lo hacen desde el terminal, utilizar el comando: `https://github.com/matirojasg/corfo-etapa-1.git`

2. Luego de obtener el código, crear un ambiente virtual dentro de la carpeta del repositorio. Esto nos permitirá instalar las librerías necesarias. Para realizar esto, usar el comando: `python3 -m venv venv`. Luego, debe ser activado de la siguiente manera: Entrar al directorio `venv` con el comando `cd venv`, luego `cd bin`, y finalmente `source ./activate`. Volver a la carpeta raíz del proyecto.

3. Instalar los requerimientos, para ello ejecutar el comando `pip install -r requirements.txt` (Esta instalación tarda un par de minutos).

4. Si quieren utilizar una GPU   NVIDIA, pueden utilizar el siguiente comando para instalar la librería Pytorch: `pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`, esto para el caso del modelo BETO, que requiere harta potencia.
Si tienen una GPU distinta, favor buscar documentación con el comando indicado para instalar Pytorch en el ambiente virtual.

Creación de archivos para entrenamiento de los modelos.
------------

1. El primer paso es colocar los archivos `reclamos_BIF_hasta_2021.txt` y `reclamos_VyS_hasta_2021.txt`, en la carpeta data/raw.

2. Ejecutar el Script: `format_data.py` ubicado en la carpeta utils. Este Script generará los archivos vys.xlsx y bif.xlsx, los cuáles quedarán ubicados en la carpeta data/cleaned. Estos archivos filtran los reclamos con menos de 5 tokens (pueden desactivar esto comentando la linea 43), y también consideran la condición impuesta en una reunión: si existe el campo mercado analista, entonces se reemplaza el valor de mercado ingreso. Además, se remueven los valores duplicados en caso que existan.

3. Crear dentro de data, una carpeta llamada input_files y ejecutar el Script: `create_input.py` ubicado en la carpeta utils. Este Script lo que hace es crear los archivos necesarios para cada una de las clasificaciones. Los archivos resultantes quedarán almacenados en la nueva carpeta creada.

4. Con respecto a los embeddings, se utilizaron los siguientes: https://github.com/crscardellino/sbwce, en la versión Word2Vec, deben ser ubicados en la carpeta embeddings (Esto no aplica para el modelo BETO, así que pueden saltarse este paso).


Entrenamiento de los modelos. (Si ya tienen los modelos pre-entrenados con todos los archivos requeridos, saltar este paso)
------------

1. Lo primero es modificar los parámetros que deseen en el archivo params.yaml. Este archivo se encuentra bien documentado para elegir correctamente los parámetros. Importante: El modelo a utilizar se elige al comienzo en el parámetro model, y más abajo en el archivo aparecen los parámetros específicos de cada modelo. Por default, están configurados los parámetros del modelo utilizado en cada una de nuestras clasificaciones finales. Además, para variar entre las distintas clasificaciones notar que hay que cambiar los campos: filepath, label y output_dir.

2. Teniendo todo configurado, el entrenamiento inicia al ejecutar el comando `python main.py`

3. Importante, el entrenamiento anterior nos entregará tres archivos fundamentales para el proceso de inferencia. En primer lugar, en la carpeta logs quedarán registrados los resultados del entrenamiento, de esta manera pueden consultar por las métricas obtenidas tanto para el conjunto de validación en cada época, como también para el conjunto de test. Segundo, en la carpeta models quedarán guardados los modelos pre-entrenados, los cuáles serán utilizados más adelante en el web service para ser consumidos. Tercero, en la carpeta src quedarán guardados los target names de cada una de las clasificaciones, es decir, todas las posibles clases asociadas a cada reclamo. Esto también es importante ya que dichas clases serán las opciones posibles al momento de realizar la inferencia de los datos sin clasificación.


Inferencia con modelos pre-entrenados.
------------

1. Situar los 10 clasificadores pre-entrenados en la carpeta models. Tener en cuenta al momento de la transferencia de archivos, que cada modelo pesa aproximadamente 400 MBs, por lo que el sistema requiere un total de 4 GBS de espacio en el disco.

2. Luego de haber cargado los modelos, subir los archivos con los target_names a la carpeta src.

3. Situar el archivo sin clasificación entrehado por la CMF en la carpeta data.

4. Finalmente, ejecutar el comando `python beto_infer.py`. Las predicciones quedarán guardadas en el archivo predictions.xlsx