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
from models import *
from scripts import benchmark_model
from transformers import *
from data import load_data
from data.base import *
from scorer import evaluate
from argparse import ArgumentParser

PRETRAINED_MODEL = None
#PRETRAINED_MODEL = PRETRAINED_LIGHTWEIGHT_CNN_TEXT_MODEL

def train(configs):
    dataset_name = configs['dataset']

    # Load dataset
    start_time = time.time()
    train, dev, test, ontology = load_data(configs['dataset'], configs['use_synthetic_train'])
    if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        train_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_train.json'))
        dev_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_dev.json'))
        test_ontology = ontology
    else:
        train_ontology = dev_ontology = test_ontology = ontology

    print('Prepared the dataset (%s seconds)'  % (time.time() - start_time))

    # Load model
    if configs['lightweight']:
        print('class LightWeightModel')
        model = LightWeightModel(configs)
    elif not configs['online_kd']:
        print('class DualBertEncodersModel')
        model = DualBertEncodersModel(configs)
    else:
        print('class EncodersModelWithOnlineKD')
        model = EncodersModelWithOnlineKD(configs)
    print('Prepared the model (Nb params: {})'.format(get_n_params(model)), flush=True)
    print(f'Nb tunable params: {get_n_tunable_params(model)}')

    # Reload a pretrained model (if exists)
    if PRETRAINED_MODEL and os.path.exists(PRETRAINED_MODEL):
        print('Reload the pretrained model')
        checkpoint = torch.load(PRETRAINED_MODEL, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Evaluate the initial model on the dev set and the test set
    print('Evaluate the initial model on the dev set and the test set')
    if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        train_ontology.build_index(model, 256)
        dev_ontology.build_index(model, 256)
        test_ontology.build_index(model, 256)
    else:
        ontology.build_index(model, 256)
    with torch.no_grad():
        if configs['hard_negatives_training']:
            train_results = evaluate(model, train, train_ontology, configs)
            print('Train results: {}'.format(train_results))
        dev_results = evaluate(model, dev, dev_ontology, configs)
        print('Dev results: {}'.format(dev_results))
        test_results = evaluate(model, test, test_ontology, configs)
        print('Test results: {}'.format(test_results))
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare the optimizer and the scheduler
    optimizer = model.get_optimizer(len(train))
    num_epoch_steps = math.ceil(len(train) / configs['batch_size'])
    print('Prepared the optimizer and the scheduler', flush=True)

    # Start Training
    accumulated_loss = RunningAverage()
    iters, batch_loss, best_dev_score, final_test_results = 0, 0, 0, None
    gradient_accumulation_steps = configs['gradient_accumulation_steps']
    for epoch_ix in range(configs['epochs']):
        print('Starting epoch {}'.format(epoch_ix+1), flush=True)
        for i in range(num_epoch_steps):
            iters += 1
            instances = train.next_items(configs['batch_size'])

            # Compute iter_loss
            iter_loss = model(instances, train_ontology, is_training=True)[0]
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

            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, round(accumulated_loss(), 3)), flush=True)
                accumulated_loss = RunningAverage()

        if (epoch_ix + 1) % configs['epoch_evaluation_frequency'] > 0: continue

        # Build the index of the ontology
        print('Starting building the index of the ontology')
        if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
            train_ontology.build_index(model, 256)
            dev_ontology.build_index(model, 256)
            test_ontology.build_index(model, 256)
        else:
            ontology.build_index(model, 256)

        # Evaluation after each epoch
        with torch.no_grad():
            if configs['hard_negatives_training']:
                train_results = evaluate(model, train, train_ontology, configs)
                print('Train results: {}'.format(train_results))

            start_dev_eval_time = time.time()
            print('Evaluation on the dev set')
            if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D] and USE_TRAINDEV:
                dev_results = evaluate(model, test, test_ontology, configs)
            else:
                dev_results = evaluate(model, dev, dev_ontology, configs)
            print(dev_results)
            dev_score = dev_results['top1_accuracy']
            print('Evaluation on the dev set took %s seconds'  % (time.time() - start_dev_eval_time))
            # if online_kd is enabled
            if configs['online_kd']:
                print('Evaluation using only the first 3 layers')
                model.enable_child_branch_exit(3)
                dev_score_3_layers = benchmark_model(model, 128, [configs['dataset']], 'dev')
                print(dev_score_3_layers)
                model.disable_child_branch_exit()

                dev_score = (dev_score + dev_score_3_layers) / 2.0

        # Save model if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print('Evaluation on the test set')
            test_results = evaluate(model, test, test_ontology, configs)
            final_test_results = test_results
            print(test_results)
            # Save the model
            save_path = join(configs['save_dir'], 'model.pt')
            torch.save({'model_state_dict': model.state_dict()}, save_path)
            print('Saved the model', flush=True)

        # Free memory of the index
        if not dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
            del ontology.namevecs_index
            ontology.namevecs_index = None
        gc.collect()
        torch.cuda.empty_cache()

    print(final_test_results)
    return final_test_results

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('-c', '--cg_config', default='lightweight_cnn_text')
    parser.add_argument('-d', '--dataset', default=BC5CDR_C, choices=DATASETS)
    args = parser.parse_args()

    # Prepare config
    configs = prepare_configs(args.cg_config, args.dataset)

    # Train
    train(configs)
