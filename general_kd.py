import os
import copy
import torch
import random
import math
import gc
import time
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from utils import *
from constants import *
from transformers import *
from data.base import Ontology
from argparse import ArgumentParser

MAX_LENGTH = 32

def encode_texts(model, tokenizer, texts, device):
    toks = tokenizer.batch_encode_plus(texts,
                                       padding=True,
                                       return_tensors='pt',
                                       truncation=True,
                                       max_length=MAX_LENGTH)
    toks = toks.to(device)
    outputs = model(**toks)
    reps = outputs[0][:, 0, :]
    return reps,

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--teacher_model', default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    parser.add_argument('--student_model', default='/shared/nas/data/m1/tuanml/biolinking/initial_models/sapbert_6_layers')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    args = parser.parse_args()
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # Save dir
    sm = args.student_model
    save_dir = sm[sm.rfind('/')+1:]
    create_dir_if_not_exist(save_dir)

    # Load teacher model and tokenizer
    teacher_model = AutoModel.from_pretrained(args.teacher_model)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.to(device)
    t_params = get_n_params(teacher_model)
    print(f'Loaded teacher model and tokenizer ({t_params} Params)')

    # Load student model
    student_model = AutoModel.from_pretrained(args.student_model)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_model.to(device)
    s_params = get_n_params(student_model)
    print(f'Loaded student model and tokenizer ({s_params} Params)')

    # Load UMLS-2017 AA Active Ontology
    ontology = Ontology(UMLS_2017AA_ACTIVE_FP)
    all_names = [n.name_str for n in ontology.name_list]
    all_names = AugmentedList(all_names, shuffle_between_epoch=True)
    print('Loaded UMLS-2017 AA Active Ontology')
    print(f'Number of names: {len(all_names)}')

    # Prepare optimizer and scheduler
    num_epoch_steps = math.ceil(len(all_names) / batch_size)
    num_train_steps = int(num_epoch_steps * epochs / gradient_accumulation_steps)
    num_warmup_steps = int(num_train_steps * 0.01)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    student_parameters = [
        {'params': [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(student_parameters, learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_steps)
    print('Prepared optimizer and scheduler')

    # Training Loop
    iters, batch_loss = 0, 0
    loss_fct = nn.MSELoss(reduction='sum')
    print('Start training')
    for epoch in range(epochs):
        accumulated_loss = RunningAverage()
        with tqdm(total=num_epoch_steps, desc=f'Epoch {epoch}') as pbar:
            for _ in range(num_epoch_steps):
                iters += 1
                batch_texts = all_names.next_items(batch_size)
                with torch.no_grad():
                    teacher_reps = encode_texts(teacher_model, teacher_tokenizer, batch_texts, device)[0]
                student_reps = encode_texts(student_model, student_tokenizer, batch_texts, device)[0]
                iter_loss = loss_fct(student_reps, teacher_reps) / batch_size
                iter_loss /= gradient_accumulation_steps
                iter_loss.backward()
                batch_loss += iter_loss.data.item()

                # Update params
                if iters % gradient_accumulation_steps == 0:
                    accumulated_loss.update(batch_loss)
                    torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    batch_loss = 0

                # Update pbar
                pbar.update(1)
                pbar.set_postfix_str(f'Iters: {iters} Student Loss: {accumulated_loss()}')

                # Save the student model and tokenizer
                if iters % 1000 == 0:
                    student_model.save_pretrained(save_dir)
                    student_tokenizer.save_pretrained(save_dir)
                    # Free any unused memory
                    gc.collect()
                    torch.cuda.empty_cache()
        print(f'Epoch {epoch} | Average Loss: {accumulated_loss()}')
