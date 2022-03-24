from cgi import test
import sys
sys.path.append('../src')
import time
import yaml
import logging
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets.datasets import ClaimDataset
from beto_training.train_beto import train
from beto_training.evaluate_beto import eval
from utils.general_utils import format_time, save_model, enable_reproducibility
from utils.beto_utils import prepare_sentences_and_labels, Loader, BetoDataset
from sklearn import preprocessing
from utils.general_utils import load_model
import re 
from sklearn.model_selection import train_test_split
import csv
import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
if __name__=='__main__':
    # Creamos un log para registrar el entrenamiento y validación del modelo.
    
    # Garantizamos reproducibilidad en nuestros experimentos.
    enable_reproducibility(seed_val = 1000)

    # Obtenemos las configuraciones del modelo beto.
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)

    filepath = config["filepath"]
    filename = filepath.split('.')[-2].split('/')[-1]
    output_labels = open(f'{filename}_target_names.txt', 'w')


    logging.basicConfig(filename=f'../logs/{filename}_log.txt', filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    beto_config = config["beto_config"]

    # Leemos el dataset de reclamos. 
    from sklearn.model_selection import KFold

    claim_dataset = ClaimDataset(filepath, pre_processing, sample = config["sample_frac"])
    
    df = claim_dataset.df

    

    # Se realiza un KFold para obtener 5 conjuntos diferentes para entrenar. 
    # La división es un 80% para el conjunto de train y 20% test.
    # Luego se extrae el 10% del conjunto de train para el conjunto de validación
    kf = KFold(n_splits = 5, shuffle = True, random_state = 2)
    train_dfs = []
    test_dfs = []
    val_dfs = []

    for train_index, test_index in kf.split(df):
        df_train = df.loc[train_index]
        df_test = df.loc[test_index]
        df_train, df_valid = train_test_split(df_train, train_size = 0.9, random_state = 2)
        train_dfs.append(df_train)
        test_dfs.append(df_test)
        val_dfs.append(df_valid)

    do_lower_case = True if beto_config["version"]=='uncased' else False
    model_name = 'dccuchile/bert-base-spanish-wwm-uncased' if do_lower_case else 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = do_lower_case)
    

    for i, (train_df, test_df, val_df) in enumerate(zip(train_dfs, test_dfs, val_dfs)):
        train_sentences, train_labels = train_df.text.values, train_df.label.values
        test_sentences, test_labels = test_df.text.values, test_df.label.values
        val_sentences, val_labels = val_df.text.values, val_df.label.values
        le = preprocessing.LabelEncoder()

        print(len(train_labels))
        print(len(test_labels))
        print(len(val_labels))
        labels = list(train_labels)+list(test_labels)+list(val_labels)
        labels = le.fit_transform(labels)
        train_labels = labels[0:len(train_labels)]
        test_labels = labels[len(train_labels): len(train_labels)+len(test_labels)]
        val_labels = labels[len(train_labels)+len(test_labels):]
        print(len(train_labels))
        print(len(test_labels))
        print(len(val_labels))
        target_names = list(le.classes_)

        train_input_ids, train_attention_masks, train_labels = prepare_sentences_and_labels(train_sentences, train_labels, tokenizer, beto_config["max_len"])
        test_input_ids, test_attention_masks, test_labels = prepare_sentences_and_labels(test_sentences, test_labels, tokenizer, beto_config["max_len"])
        val_input_ids, val_attention_masks, val_labels = prepare_sentences_and_labels(val_sentences, val_labels, tokenizer, beto_config["max_len"])
    
        train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
        test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
        val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

        batch_size = beto_config["batch_size"]
        train_dataloader, validation_dataloader, testing_dataloader = Loader(train_dataset, val_dataset, test_dataset, batch_size).create_data_loader()
        
        model = BertForSequenceClassification.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-uncased' if do_lower_case else 'dccuchile/bert-base-spanish-wwm-cased', 
            num_labels = len(target_names), 
            output_attentions = False, 
            output_hidden_states = False, 
        )

        # Parámetros del modelo a la gpu en caso de que se especifique en las configuraciones.
        if device == 'cuda':
            model.cuda()
    
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': beto_config["weight_decay_rate"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]


        optimizer = AdamW(optimizer_grouped_parameters, 
                        lr = float(beto_config["lr"]), 
                        eps = float(beto_config["eps"])
                        )
    
        epochs = beto_config["epochs"]

        total_steps = len(train_dataloader) * epochs

    
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = beto_config["num_warmup_steps"], 
                                                num_training_steps = total_steps)


        total_t0 = time.time()
        training_stats = []

        for epoch_i in range(0, epochs):
            print('======== Época {:} / {:} ========'.format(epoch_i + 1, epochs))
            avg_train_loss, training_time = train(labels, model, optimizer, scheduler, train_dataloader, device)
            avg_val_loss, f1, validation_time = eval(model, validation_dataloader, device, 'val')
            training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. F1 .': f1,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
            )
    
    
    
        print("Entrenamiento completo!")

        print(f"El entrenamiento tomó un total de {format_time(time.time()-total_t0)} segundos.")

        print("================================================")
        print("")
        
        print("Probando el modelo en el conjunto de Testing .....")

        save_model(model, tokenizer, beto_config["output_dir"])
        bert_model = load_model(model, tokenizer, beto_config["output_dir"], 'cuda')
        eval(bert_model, testing_dataloader, device, 'test')