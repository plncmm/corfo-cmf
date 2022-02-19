import sys
sys.path.append('../src')
import time
import yaml
import logging
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets.datasets import ClaimDataset
from beto_training.train_beto import train
from beto_training.evaluate_beto import eval
from utils.general_utils import format_time, save_plot, save_model, enable_reproducibility
from utils.beto_utils import prepare_sentences_and_labels, Loader, BetoDataset
from sklearn import preprocessing

if __name__=='__main__':
    # Creamos un log para registrar el entrenamiento y validación del modelo.
    logging.basicConfig(filename='../logs/beto_log.txt', filemode='w', format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    
    # Garantizamos reproducibilidad en nuestros experimentos.
    enable_reproducibility(seed_val = 1000)

    # Obtenemos las configuraciones del modelo beto.
    with open('../params.yaml') as file:
        config = yaml.safe_load(file)

    filepath = config["filepath"]
    model = config["model"]
    device = config["device"]
    pre_processing = config["pre_processing"]
    beto_config = config["beto_config"]

    # Leemos el dataset de reclamos. 

    claim_dataset = ClaimDataset(filepath, pre_processing, sample = config["sample_frac"])
    
    df = claim_dataset.df

    sentences, labels = df.text.values, df.label.values

    
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    target_names = list(le.classes_)
    
    do_lower_case = True if beto_config["version"]=='uncased' else False
    model_name = 'dccuchile/bert-base-spanish-wwm-uncased' if do_lower_case else 'dccuchile/bert-base-spanish-wwm-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case = do_lower_case)
    input_ids, attention_masks, labels = prepare_sentences_and_labels(sentences, labels, tokenizer, beto_config["max_len"])
    train_dataset, val_dataset, test_dataset = BetoDataset(input_ids, attention_masks, labels).create_partitions(train_size = beto_config["train_size"], test_size = beto_config["test_size"], val_size = beto_config["val_size"])
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

    eval(model, testing_dataloader, device, 'test')

    save_model(model, tokenizer, beto_config["output_dir"])
