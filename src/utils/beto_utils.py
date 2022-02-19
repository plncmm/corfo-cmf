import torch 
import time
import math
from tqdm import tqdm
from torch.utils.data import random_split, TensorDataset, DataLoader, RandomSampler, SequentialSampler



def prepare_sentences_and_labels(sentences, labels, tokenizer, max_len=512):
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
                            max_length = max_len,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                    )
            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

    # Convertimos a tensores de Pytorch.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

class BetoDataset:
    def __init__(self, input_ids, attention_masks, labels) -> None:
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels 

    def create_partitions(self, train_size = 0.8, test_size = 0.1, val_size = 0.1):
        # To do: Arreglar el math.ceil, está muy a mano
        dataset = TensorDataset(self.input_ids, self.attention_masks, self.labels)
        train_size = math.ceil(int(train_size * len(dataset)))
        test_size = math.ceil(test_size * len(dataset))
        val_size = len(dataset) - train_size - test_size
        train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator=torch.Generator().manual_seed(42))
        return train_dataset, val_dataset, test_dataset
    


class Loader:
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size) -> None:
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
    
    def create_data_loader(self):
        
        # Oversample the minority classes
        
   
        train_dataloader = DataLoader(
                        self.train_dataset,  
                        sampler = RandomSampler(self.train_dataset), 
                        batch_size = self.batch_size 
                    )

            
        validation_dataloader = DataLoader(
                    self.val_dataset, 
                    sampler = SequentialSampler(self.val_dataset), 
                    batch_size = self.batch_size 
                )
        
        testing_dataloader = DataLoader(
                    self.test_dataset, 
                    sampler = SequentialSampler(self.test_dataset), 
                    batch_size = self.batch_size 
                )
        return train_dataloader, validation_dataloader, testing_dataloader