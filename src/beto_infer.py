from unicodedata import name
import torch
import pandas as pd
import re
import csv
import numpy as np
from utils.general_utils import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time 

torch.cuda.set_device(1)

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
    t0 = time.time()
    df = get_df('../data/datos_sinClasificacion.csv')
    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased', do_lower_case = True)
    output = open('predictions.txt', 'w')
    
    model = BertForSequenceClassification.from_pretrained(
        'dccuchile/bert-base-spanish-wwm-uncased', 
        num_labels = 2, 
        output_attentions = False, 
        output_hidden_states = False, 
    )

    model.to('cuda')

    # Tipo Mercado

    tipo_mercado = load_model(model, tokenizer, '../models/tipo_mercado', 'cuda')
    tipo_mercado.to('cuda')
    tipo_mercado.eval()

    # Tipo Mercado Seguros o Valores

    tipo_mercado_seguros_valores = load_model(model, tokenizer, '../models/tipo_mercado_seguros_valores', 'cuda')
    tipo_mercado_seguros_valores.to('cuda')
    tipo_mercado_seguros_valores.eval()

    # Bancos Tipo Entidad

    bancos_tipo_entidad = load_model(model, tokenizer, '../models/bancos_tipo_entidad', 'cuda')
    bancos_tipo_entidad.to('cuda')
    bancos_tipo_entidad.eval()

    # Bancos Tipo Producto

    bancos_tipo_producto = load_model(model, tokenizer, '../models/bancos_tipo_producto', 'cuda')
    bancos_tipo_producto.to('cuda')
    bancos_tipo_producto.eval()

    # Bancos Tipo Materia

    bancos_tipo_materia = load_model(model, tokenizer, '../models/bancos_tipo_materia', 'cuda')
    bancos_tipo_materia.to('cuda')
    bancos_tipo_materia.eval()

    # Bancos Nombre Entidad

    bancos_nombre_entidad = load_model(model, tokenizer, '../models/bancos_nombre_entidad', 'cuda')
    bancos_nombre_entidad.to('cuda')
    bancos_nombre_entidad.eval()



    # SyV Tipo Entidad

    seguros_valores_tipo_entidad = load_model(model, tokenizer, '../models/seguros_valores_tipo_entidad', 'cuda')
    seguros_valores_tipo_entidad.to('cuda')
    seguros_valores_tipo_entidad.eval()

    # seguros_valores Tipo Producto

    seguros_valores_tipo_producto = load_model(model, tokenizer, '../models/seguros_valores_tipo_producto', 'cuda')
    seguros_valores_tipo_producto.to('cuda')
    seguros_valores_tipo_producto.eval()

    # seguros_valores Tipo Materia

    seguros_valores_tipo_materia = load_model(model, tokenizer, '../models/seguros_valores_tipo_materia', 'cuda')
    seguros_valores_tipo_materia.to('cuda')
    seguros_valores_tipo_materia.eval()

    # seguros_valores Nombre Entidad

    seguros_valores_nombre_entidad = load_model(model, tokenizer, '../models/seguros_valores_nombre_entidad', 'cuda')
    seguros_valores_nombre_entidad.to('cuda')
    seguros_valores_nombre_entidad.eval()



    sentences = df.Reclamo.values
    

    input_ids = []
    attention_masks = []


    for sent in sentences:
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
        
  
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)


    # Set the batch size.  
    batch_size = 1  

    # Create the DataLoader.
    prediction_data = TensorDataset(input_ids, attention_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    


    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    # Put model in evaluation mode
    

    device = 'cuda'

    # Tracking variables 
    predictions , true_labels = [], []

    # Predict 

    table = []

    
    df['MERCADO_INGRESO'] = ''
    df['TIPO_ENTIDAD'] = ''
    df['NOMBRE_ENTIDAD'] = ''
    df['TIPO_PRODUCTO'] = ''
    df['TIPO_MATERIA'] = '' 

    for i, batch in enumerate(prediction_dataloader):
        sentence_data = []
        
        if i%100==0:
            print(f'Sentence number: {i}')
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
      
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            
            outputs_tipo_mercado = tipo_mercado(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

            outputs_tipo_mercado_seguros_valores = tipo_mercado_seguros_valores(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

            outputs_bancos_tipo_producto = bancos_tipo_producto(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_bancos_tipo_entidad = bancos_tipo_entidad(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_bancos_tipo_materia = bancos_tipo_materia(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_bancos_nombre_entidad = bancos_nombre_entidad(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

            
            outputs_seguros_valores_tipo_producto = seguros_valores_tipo_producto(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_seguros_valores_tipo_entidad = seguros_valores_tipo_entidad(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_seguros_valores_tipo_materia = seguros_valores_tipo_materia(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)
            
            outputs_seguros_valores_nombre_entidad = seguros_valores_nombre_entidad(b_input_ids, token_type_ids=None, 
                            attention_mask=b_input_mask)

            

        logits_tipo_mercado = outputs_tipo_mercado[0]
        logits_tipo_mercado_seguros_valores = outputs_tipo_mercado_seguros_valores[0]

        logits_bancos_tipo_entidad = outputs_bancos_tipo_entidad[0]
        logits_bancos_tipo_producto = outputs_bancos_tipo_producto[0]
        logits_bancos_tipo_materia = outputs_bancos_tipo_materia[0]
        logits_bancos_nombre_entidad = outputs_bancos_nombre_entidad[0]
        
        logits_seguros_valores_tipo_entidad = outputs_seguros_valores_tipo_entidad[0]
        logits_seguros_valores_tipo_producto = outputs_seguros_valores_tipo_producto[0]
        logits_seguros_valores_tipo_materia = outputs_seguros_valores_tipo_materia[0]
        logits_seguros_valores_nombre_entidad = outputs_seguros_valores_nombre_entidad[0]


        # Move logits and labels to CPU
        logits_tipo_mercado = logits_tipo_mercado.detach().cpu().numpy()
        pred_tipo_mercado = np.argmax(logits_tipo_mercado, axis=1).flatten()

        # Move logits and labels to CPU
        logits_tipo_mercado_seguros_valores = logits_tipo_mercado_seguros_valores.detach().cpu().numpy()
        pred_tipo_mercado_seguros_valores = np.argmax(logits_tipo_mercado_seguros_valores, axis=1).flatten()


        # Move logits and labels to CPU
        logits_bancos_tipo_entidad = logits_bancos_tipo_entidad.detach().cpu().numpy()
        pred_bancos_tipo_entidad = np.argmax(logits_bancos_tipo_entidad, axis=1).flatten()


        # Move logits and labels to CPU
        logits_bancos_tipo_producto = logits_bancos_tipo_producto.detach().cpu().numpy()
        pred_bancos_tipo_producto = np.argmax(logits_bancos_tipo_producto, axis=1).flatten()

        # Move logits and labels to CPU
        logits_bancos_tipo_materia = logits_bancos_tipo_materia.detach().cpu().numpy()
        pred_bancos_tipo_materia = np.argmax(logits_bancos_tipo_materia, axis=1).flatten()

        # Move logits and labels to CPU
        logits_bancos_nombre_entidad = logits_bancos_nombre_entidad.detach().cpu().numpy()
        pred_bancos_nombre_entidad = np.argmax(logits_bancos_nombre_entidad, axis=1).flatten()


        # Move logits and labels to CPU
        logits_seguros_valores_tipo_entidad = logits_seguros_valores_tipo_entidad.detach().cpu().numpy()
        pred_seguros_valores_tipo_entidad = np.argmax(logits_seguros_valores_tipo_entidad, axis=1).flatten()


        # Move logits and labels to CPU
        logits_seguros_valores_tipo_producto = logits_seguros_valores_tipo_producto.detach().cpu().numpy()
        pred_seguros_valores_tipo_producto = np.argmax(logits_seguros_valores_tipo_producto, axis=1).flatten()

        # Move logits and labels to CPU
        logits_seguros_valores_tipo_materia = logits_seguros_valores_tipo_materia.detach().cpu().numpy()
        pred_seguros_valores_tipo_materia = np.argmax(logits_seguros_valores_tipo_materia, axis=1).flatten()

        # Move logits and labels to CPU
        logits_seguros_valores_nombre_entidad = logits_seguros_valores_nombre_entidad.detach().cpu().numpy()
        pred_seguros_valores_nombre_entidad = np.argmax(logits_seguros_valores_nombre_entidad, axis=1).flatten()
        

        output.write(sentences[i]+'\n')
        
        # Tipo Mercado

        tipo_mercado_opciones = open('tipo_mercado_target_names.txt', 'r').read().splitlines()
        tipo_mercado_opciones = {v: k for v, k in enumerate(tipo_mercado_opciones)}
        
        
        
       
        sentence_data.append(sentences[i])
        
        if int(pred_tipo_mercado)==0:
            output.write('Tipo de mercado: ' + tipo_mercado_opciones[int(pred_tipo_mercado)] + '\n')
            sentence_data.append(tipo_mercado_opciones[int(pred_tipo_mercado)])
            df.at[i,'MERCADO_INGRESO'] = tipo_mercado_opciones[int(pred_tipo_mercado)]

            # Bancos Tipo entidad
            bancos_tipo_entidad_opciones = open('bancos_tipo_entidad_target_names.txt', 'r').read().splitlines()
            bancos_tipo_entidad_opciones = {v: k for v, k in enumerate(bancos_tipo_entidad_opciones)}
            output.write('Tipo de entidad: ' + bancos_tipo_entidad_opciones[int(pred_bancos_tipo_entidad)] + '\n')
            sentence_data.append(bancos_tipo_entidad_opciones[int(pred_bancos_tipo_entidad)])
            df.at[i,'TIPO_ENTIDAD'] = bancos_tipo_entidad_opciones[int(pred_bancos_tipo_entidad)]

            # Bancos nombre Entidad
            bancos_nombre_entidad_opciones = open('bancos_nombre_entidad_target_names.txt', 'r').read().splitlines()
            bancos_nombre_entidad_opciones = {v: k for v, k in enumerate(bancos_nombre_entidad_opciones)}
            output.write('Nombre de entidad: ' + bancos_nombre_entidad_opciones[int(pred_bancos_nombre_entidad)] + '\n')
            sentence_data.append(bancos_nombre_entidad_opciones[int(pred_bancos_nombre_entidad)])
            df.at[i,'NOMBRE_ENTIDAD'] = bancos_nombre_entidad_opciones[int(pred_bancos_nombre_entidad)]

            # Bancos tipo materia
            bancos_tipo_materia_opciones = open('bancos_tipo_materia_target_names.txt', 'r').read().splitlines()
            bancos_tipo_materia_opciones = {v: k for v, k in enumerate(bancos_tipo_materia_opciones)}
            output.write('Tipo de materia: ' + bancos_tipo_materia_opciones[int(pred_bancos_tipo_materia)] + '\n')
            sentence_data.append(bancos_tipo_materia_opciones[int(pred_bancos_tipo_materia)])
            df.at[i,'TIPO_MATERIA'] = bancos_tipo_materia_opciones[int(pred_bancos_tipo_materia)]

            # Bancos tipo producto
            bancos_tipo_producto_opciones = open('bancos_tipo_producto_target_names.txt', 'r').read().splitlines()
            bancos_tipo_producto_opciones = {v: k for v, k in enumerate(bancos_tipo_producto_opciones)}
            output.write('Tipo de producto: ' + bancos_tipo_producto_opciones[int(pred_bancos_tipo_producto)] + '\n')
            sentence_data.append(bancos_tipo_producto_opciones[int(pred_bancos_tipo_producto)])
            df.at[i,'TIPO_PRODUCTO'] = bancos_tipo_producto_opciones[int(pred_bancos_tipo_producto)]

        else:
            
            
            tipo_mercado_seguros_valores_opciones = open('tipo_mercado_seguros_valores_target_names.txt', 'r').read().splitlines()
            tipo_mercado_seguros_valores_opciones = {v: k for v, k in enumerate(tipo_mercado_seguros_valores_opciones)}
            #output.write('Tipo de mercado seguros o valores: ' + tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)] + '\n')
            output.write('Tipo de mercado: ' + tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)] + '\n')
            sentence_data.append(tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)])
            df.at[i,'MERCADO_INGRESO'] = tipo_mercado_seguros_valores_opciones[int(pred_tipo_mercado_seguros_valores)]



            # Seguros Valores Tipo entidad
            seguros_valores_tipo_entidad_opciones = open('seguros_valores_tipo_entidad_target_names.txt', 'r').read().splitlines()
            seguros_valores_tipo_entidad_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_entidad_opciones)}
            output.write('Tipo de entidad: ' + seguros_valores_tipo_entidad_opciones[int(pred_seguros_valores_tipo_entidad)] + '\n')
            sentence_data.append(seguros_valores_tipo_entidad_opciones[int(pred_seguros_valores_tipo_entidad)])
            df.at[i,'TIPO_ENTIDAD'] = seguros_valores_tipo_entidad_opciones[int(pred_seguros_valores_tipo_entidad)]

            # Seguros Valores nombre Entidad
            seguros_valores_nombre_entidad_opciones = open('seguros_valores_nombre_entidad_target_names.txt', 'r').read().splitlines()
            seguros_valores_nombre_entidad_opciones = {v: k for v, k in enumerate(seguros_valores_nombre_entidad_opciones)}
            output.write('Nombre entidad: ' + seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)] + '\n')
            sentence_data.append(seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)])
            df.at[i,'NOMBRE_ENTIDAD'] = seguros_valores_nombre_entidad_opciones[int(pred_seguros_valores_nombre_entidad)]

            # Seguros Valores tipo materia
            seguros_valores_tipo_materia_opciones = open('seguros_valores_tipo_materia_target_names.txt', 'r').read().splitlines()
            seguros_valores_tipo_materia_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_materia_opciones)}
            output.write('Tipo materia: ' + seguros_valores_tipo_materia_opciones[int(pred_seguros_valores_tipo_materia)] + '\n')
            sentence_data.append(seguros_valores_tipo_materia_opciones[int(pred_seguros_valores_tipo_materia)])
            df.at[i,'TIPO_MATERIA'] = seguros_valores_tipo_materia_opciones[int(pred_seguros_valores_tipo_materia)]

            # Seguros Valores tipo producto
            seguros_valores_tipo_producto_opciones = open('seguros_valores_tipo_producto_target_names.txt', 'r').read().splitlines()
            seguros_valores_tipo_producto_opciones = {v: k for v, k in enumerate(seguros_valores_tipo_producto_opciones)}
            output.write('Tipo producto: ' + seguros_valores_tipo_producto_opciones[int(pred_seguros_valores_tipo_producto)] + '\n')
            sentence_data.append(seguros_valores_tipo_producto_opciones[int(pred_seguros_valores_tipo_producto)])
            df.at[i,'TIPO_PRODUCTO'] = seguros_valores_tipo_producto_opciones[int(pred_seguros_valores_tipo_producto)]

        table.append(sentence_data)
        output.write('\n')
        
    del df['Reclamo']
    df.to_excel('predictions.xlsx')
    output.close()
    print(time.time()-t0)
    print('    DONE.')
