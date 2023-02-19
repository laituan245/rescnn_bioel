import os
import copy
import torch
import random
import json
import math
import gc
import time
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils import prepare_configs, get_n_params, create_dir_if_not_exist, get_n_tunable_params, AugmentedList, RunningAverage
from models import DualBertEncodersModel
from constants import PRETRAINING_MODEL, UMLS_2020AA_FULL_FP, UMLS_PRETRAIN_POSITIVE_PAIRS
from constants import BC5CDR_C, BC5CDR_D, NCBI_D, COMETA
from data.base import Ontology, DataInstance, PretrainingPositivePairs
from argparse import ArgumentParser
from scripts import benchmark_model

def pretrain(configs):
    # Load model
    model = DualBertEncodersModel(configs)
    model.model_type = PRETRAINING_MODEL
    save_path = os.path.join(configs['save_dir'], 'model.pt')
    print(f'Prepared the model (Params: {get_n_params(model)})')
    print(f'Nb Tunable Params: {get_n_tunable_params(model)}')
    print(f'Save path {save_path}')

    # Load UMLS-2020 AA Full Ontology
    ontology = Ontology(UMLS_2020AA_FULL_FP)
    print('Loaded UMLS-2020 AA Full Ontology')

    # Prepare the train set
    train, example_id = [], 0
    positive_pairs = PretrainingPositivePairs(UMLS_PRETRAIN_POSITIVE_PAIRS)
    for n1, n2 in positive_pairs:
        mention = {
            'term': n1.name_str,
            'entity_id': n1.entity_id
        }
        inst = DataInstance(example_id, '', mention)
        inst.selected_positive = n2
        train.append(inst)
        example_id += 1
    random.shuffle(train)
    train = AugmentedList(train, shuffle_between_epoch=True)
    print(f'Train size: {len(train)}')


    # Prepare the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    num_epoch_steps = math.ceil(len(train) / configs['batch_size'])
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start Training
    print('Batch size: {}'.format(configs['batch_size']))
    print('Epochs: {}'.format(configs['epochs']))
    iters, batch_loss, best_score = 0, 0, 0
    batch_size = configs['batch_size']
    accumulated_loss = RunningAverage()
    gradient_accumulation_steps = configs['gradient_accumulation_steps']
    for epoch in range(configs['epochs']):
        with tqdm(total=num_epoch_steps, desc=f'Epoch {epoch}') as pbar:
            for _ in range(num_epoch_steps):
                iters += 1
                instances = train.next_items(batch_size)

                # Compute iter_loss
                iter_loss = model(instances, ontology, is_training=True)[0]
                iter_loss = iter_loss / gradient_accumulation_steps
                iter_loss.backward()
                batch_loss += iter_loss.data.item()

                # Update params
                if iters % gradient_accumulation_steps == 0:
                    accumulated_loss.update(batch_loss)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), configs['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = 0

                # Update pbar
                pbar.update(1)
                pbar.set_postfix_str(f'Iters: {iters} Student Loss: {accumulated_loss()}')

                # Evaluation and Model Saving
                if (iters % 5000 == 0) or (iters % num_epoch_steps == 0):
                    print(f'{iters} Benchmarking model')
                    model_score = benchmark_model(model, batch_size, [BC5CDR_C, BC5CDR_D, NCBI_D, COMETA], 'test')
                    print('Overall model score: {}'.format(model_score))
                    if model_score > best_score:
                        best_score = model_score
                        torch.save({'model_state_dict': model.state_dict()}, save_path)
                        adapter_dir = '{}-umls-synonyms'.format(configs['transformer'])
                        if '/' in adapter_dir:
                            adapter_dir = adapter_dir.split('/')[-1]
                        create_dir_if_not_exist(adapter_dir)
                        model.transformer.save_adapter(adapter_dir, 'umls-synonyms')
                        print('Saved a new ckpt')

                    print('', flush=True)

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default='cg_basic_pretraining_v2')
    parser.add_argument('--transformer', default=None)
    args = parser.parse_args()

    # Prepare config
    configs = prepare_configs(
        config_name=args.config,
        dataset='UMLS-2020AA-Full',
        transformer=args.transformer
    )

    # Train
    pretrain(configs)
