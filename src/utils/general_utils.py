import torch
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import random
import numpy as np

def get_device():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print('GPU disponible: {}'.format(torch.cuda.get_device_name(0)))
        device = torch.device('cuda')
    else:
        print('No hay GPU disponible, utilizaremos la CPU.')
        device = torch.device('cpu')
    return device

def enable_reproducibility(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def save_plot(training_stats):
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])
    plt.savefig('curves.png')
    
def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Guardando el modelo en: {output_dir}")
    model_to_save = model.module if hasattr(model, 'module') else model  
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
def load_model(model, tokenizer, dir, device):
    print("Cargando el modelo ...")
    model = model.from_pretrained(dir)
    tokenizer = tokenizer.from_pretrained(dir)
    model.to(device)
    return model