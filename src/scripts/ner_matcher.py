from cgi import test
from unicodedata import name
import torch
import pandas as pd
import re
import csv
import numpy as np
from utils.general_utils import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils.beto_utils import prepare_sentences_and_labels, Loader, BetoDataset
import time 
from tqdm import tqdm 
from flair.data import Sentence
from flair.models import SequenceTagger
import difflib 
from utils.general_utils import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import f1_score
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
torch.cuda.set_device(2)

def get_df(path):
    #input_file = open(path, 'r', encoding='latin-1').read() # Archivo con información sobre Bancos.

    import pandas as pd

    df = pd.read_csv(path, delimiter=';')

    df = df.iloc[:, 0:-1]

    print(df.shape)
    print(df.head())
    
    df["Reclamo"] = df['DESCRIPCION_PROBLEMA'] + '. ' + df['PETICION_SOLICITUD'] 
    df = df[df['Reclamo'].str.strip()!='']
    
    
    return df

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case = True)
    model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-uncased', 
        num_labels = 2, 
        output_attentions = False, 
        output_hidden_states = False, 
    )

    df = get_df('../data/datos_sinClasificacion.csv')

    model.to('cuda')
    real_df = pd.read_excel('../data/input_files/seguros_valores_nombre_entidad.xlsx', usecols = ["Reclamo", "NOMBRE_ENTIDAD"])
    real_labels = real_df.NOMBRE_ENTIDAD.values
    
    
    sentences = df.Reclamo.values[0:100]

    input_ids = []
    attention_masks = []
    
    print("Tokenizando las oraciones.")
    #Xtotal_sentences = len(sentences)

    with tqdm(sentences, unit="sentences") as pbar:
    
        for sent in pbar:
            # Con esta función se llevan a cabo las siguientes operaciones: 1) Se tokeniza la oración. 2) Se agregan los tokens especiales [CLS] y [SEP]. 3) Se mapean
            # los tokens a los IDs del vocabulario de BETO. 4) Se realiza un padding sobre el largo máximo establecido. 6) Se crea la máscara de atención que indica que 
            # tokens corresponden a un padding y cuáles deben ser atendidos.
            encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 512,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

    # Convertimos a tensores de Pytorch.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    
    
    seguros_valores_nombre_entidad_opciones = open('seguros_valores_nombre_entidad_target_names.txt', 'r').read().splitlines()
    seguros_valores_nombre_entidad_opciones = {v: k for v, k in enumerate(seguros_valores_nombre_entidad_opciones)}
    

    # Tipo Mercado

    seguros_valores_nombre_entidad = load_model(model, tokenizer, '../models/seguros_valores_nombre_entidad', 'cuda')
    seguros_valores_nombre_entidad.to('cuda')
    seguros_valores_nombre_entidad.eval()

    

    

   

    t0 = time.time()
    
    print('Loading tagger')
    # load tagger
    tagger = SequenceTagger.load("flair/ner-spanish-large")

    print('Tagger ready.')

    
    pred = []

    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(
                    dataset, 
                    sampler = SequentialSampler(dataset), 
                    batch_size = 1
                )

    batches = []
    for i, batch in enumerate(dataloader):
        batches.append(batch)

    with tqdm(sentences, unit="sentences") as pbar:
    
        for i, claim in enumerate(pbar):
           
            sentence = Sentence(claim)
           
            tagger.predict(sentence)

            org = ""
            org_found = False
            for entity in sentence.get_spans('ner'):
                if entity.tag=='ORG':
                    print(entity.text)
                    #a = difflib.get_close_matches(entity.text, real_labels, n=1, cutoff=0.6)
                    if any(entity.text.lower()==word.lower() for word in real_labels):  
                        org = entity.text.lower() + ": Matcher"
                        org_found = True
                        print('here')
                        break

            if not org_found: 
                with torch.no_grad():
                    input_ids = batches[i][0].cuda()
                    input_mask = batches[i][1].cuda()
                    outputs_seguros_valores_nombre_entidad = seguros_valores_nombre_entidad(input_ids, token_type_ids=None, 
                                attention_mask=input_mask)


                logits_seguros_valores_nombre_entidad = outputs_seguros_valores_nombre_entidad[0]

                logits_seguros_valores_nombre_entidad = logits_seguros_valores_nombre_entidad.detach().cpu().numpy()
                pred_seguros_valores_nombre_entidad = np.argmax(logits_seguros_valores_nombre_entidad, axis=1).flatten()
                org = seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)].lower()
            
            pred.append(org)
        


    output = open('predictions.txt', 'w')

    for i, p in enumerate(pred):
        output.write(f"Sentence: {sentences[i]}\n\n")
        output.write(f"Predicted: {p}\n")
        output.write('\n\n')

    output.close()

   