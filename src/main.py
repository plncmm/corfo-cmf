import yaml
import torch
from subprocess import call


with open('../params.yaml') as file:   # Abrimos el archivo con las configuraciones del experimento.
    config = yaml.safe_load(file)

model_name = config["model"]        # Modelo a utilizar.

print(f'Ejecutando el modelo "{model_name}" con los siguientes argumentos:')
print('')

for k, v in config[f"{model_name}_config"].items():
    print(f'\t{k}: {v}')

print('')

if model_name in ('beto', 'roberta', 'xlnet') and config["device"] == 'cuda': # Si es que trabajamos con modelos de la librería transformers, utilizamos cuda.
    torch.cuda.set_device(config["cuda_id"])

call(["python", f"models/run_{model_name}.py"])    # Ejecutamos el Script correspondiente al modelo.