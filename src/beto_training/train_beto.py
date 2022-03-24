import time 
import torch 
import logging 
import re 
from tqdm import tqdm
from utils.general_utils import format_time 
from sadice import SelfAdjDiceLoss
from sklearn.utils import class_weight
import torch.nn as nn
import numpy as np



def train(y, model, optimizer, scheduler, train_dataloader, device):# Training steps: batches * epochs
    print("")
    #criterion = SelfAdjDiceLoss()
    #class_weights=class_weight.compute_class_weight(class_weight ='balanced',classes = np.unique(y),y = y.numpy())
    #class_weights=torch.tensor(class_weights,dtype=torch.float)
    #criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    total_train_loss = 0
    model.train()
    with tqdm(train_dataloader, unit="batches") as pbar:
    
        for batch in pbar:

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            loss, logits = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)[:2]

            loss = criterion(logits.cpu(), b_labels.cpu()) # Borrar esto
          
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

 
    avg_train_loss = total_train_loss / len(train_dataloader)            
    training_time = format_time(time.time() - t0)


    print("")
    print(f"  Loss en el conjunto de entrenamiento: {avg_train_loss}")
    print(f"  Tiempo de entrenamiento: {training_time}")

    logging.info("")
    logging.info(f"  Loss en el conjunto de entrenamiento: {avg_train_loss}")
    logging.info(f"  Tiempo de entrenamiento: {training_time}")
    
    return avg_train_loss, training_time