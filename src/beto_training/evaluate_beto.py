import torch 
import time 
import numpy as np
import logging
from tqdm import tqdm
from utils.general_utils import format_time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, f1_score

def eval(model, dataloader, device, type = 'val'):
    t0 = time.time()
    model.eval()
    print("")
    print(f"Evaluando.....")
    logging.info("")
    logging.info(f"Evaluando.....")
    total_eval_loss = 0
    y_pred = []
    y_true = []


    with tqdm(dataloader, unit="batches") as pbar:
        for batch in pbar:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
    
            with torch.no_grad():        

                output = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask,
                                    labels=b_labels)
                
                loss = output['loss']
                logits = output['logits']
                
            total_eval_loss += loss.item()
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            pred = np.argmax(logits, axis=1).flatten()
            true = label_ids.flatten()
            y_pred.extend(pred)
            y_true.extend(true)

   

    f1 = f1_score(y_true, y_pred, average='micro')
    if type == 'test': 
        print(classification_report(y_true, y_pred))
        logging.info(classification_report(y_true, y_pred))
        p = precision_score(y_true, y_pred, average='weighted')
        r = recall_score(y_true, y_pred, average='weighted')
        f = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {acc}')
        print(f'Precision: {p}')
        print(f'Recall: {r}')
        print(f"Weighted F1-Score en conjunto de test: {f}")

        logging.info(f'Accuracy: {acc}')
        logging.info(f'Precision: {p}')
        logging.info(f'Recall: {r}')
        logging.info(f"Weighted F1-Score en conjunto de test: {f}")
        
    if type == 'val': 
        
        p = precision_score(y_true, y_pred, average='weighted')
        r = recall_score(y_true, y_pred, average='weighted')
        f = f1_score(y_true, y_pred, average='weighted')
        acc = accuracy_score(y_true, y_pred)
        print(f'Accuracy: {acc}')
        print(f'Precision: {p}')
        print(f'Recall: {r}')
        print(f"Weighted F1-Score en conjunto de validación: {f}")
        logging.info(f'Accuracy: {acc}')
        logging.info(f'Precision: {p}')
        logging.info(f'Recall: {r}')
        logging.info(f"Weighted F1-Score en conjunto de validación: {f}")
        
        print(classification_report(y_true, y_pred))
        logging.info(classification_report(y_true, y_pred))

    avg_val_loss = total_eval_loss / len(dataloader)
    validation_time = format_time(time.time() - t0)
    print(f"Loss en el conjunto de validación: {avg_val_loss}")
    print("")
    logging.info(f"Loss en el conjunto de validación: {avg_val_loss}")
    logging.info("")
    

        
    return avg_val_loss, f1, validation_time

    