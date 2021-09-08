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
from transformers import *
from data import load_data
from data.base import *
from scorer import evaluate
from argparse import ArgumentParser

PRETRAINED_MODEL = None

def train_reranker(cg_configs, reranker_configs):
    assert(cg_configs['dataset'] == reranker_configs['dataset'])
    dataset_name = reranker_configs['dataset']

    # Load dataset
    start_time = time.time()
    train, dev, test, ontology = load_data(dataset_name)
    if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        train_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_train.json'))
        dev_ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_dev.json'))
        test_ontology = ontology
    else:
        train_ontology = dev_ontology = test_ontology = ontology
    print('Prepared the dataset (%s seconds)'  % (time.time() - start_time))

    # Reload the candidate generator
    if not cg_configs['online_kd']:
        candidate_generator = DualBertEncodersModel(cg_configs)
    else:
        candidate_generator = EncodersModelWithOnlineKD(cg_configs)
    cg_trained_path = cg_configs['trained_path']
    assert(os.path.exists(cg_trained_path))
    if os.path.exists(cg_trained_path):
        checkpoint = torch.load(cg_trained_path, map_location=candidate_generator.device)
        candidate_generator.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print('Reloaded the candidate generator ({} params)'.format(get_n_params(candidate_generator)))

    # Build the ontology
    print('Building the ontology(s)')
    if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        train_ontology.build_index(candidate_generator, 256)
        dev_ontology.build_index(candidate_generator, 256)
        test_ontology.build_index(candidate_generator, 256)
    else:
        ontology.build_index(candidate_generator, 256)

    # Re-evaluate the candidate generator
    print('Re-evaluate the candidate generator')
    with torch.no_grad():
        train_results = evaluate(candidate_generator, train, train_ontology, cg_configs)
        print(train_results)
        dev_results = evaluate(candidate_generator, dev, dev_ontology, cg_configs)
        print(dev_results)
        test_results = evaluate(candidate_generator, test, test_ontology, cg_configs)
        print(test_results)

    # Free memory
    candidate_generator.to(torch.device('cpu'))
    del candidate_generator
    del ontology.namevecs_index
    ontology.namevecs_index = None
    gc.collect()
    torch.cuda.empty_cache()

    # Initialize a new reranker
    # Prepare its optimizer
    reranker = CrossAttentionEncoderModel(reranker_configs)
    optimizer = reranker.get_optimizer(len(train))
    num_epoch_steps = math.ceil(len(train) / reranker_configs['batch_size'])
    print('Initialized a new reranker ({} params)'.format(get_n_params(reranker)))

    # Evaluate the new reranker
    print('Evaluate the new reranker')
    with torch.no_grad():
        dev_results = evaluate(reranker, dev, dev_ontology, reranker_configs)
        print(dev_results)
        test_results = evaluate(reranker, test, test_ontology, reranker_configs)
        print(test_results)

    # Start Training
    configs = reranker_configs
    accumulated_loss = RunningAverage()
    iters, batch_loss, best_dev_score, final_test_results = 0, 0, 0, None
    gradient_accumulation_steps = configs['gradient_accumulation_steps']
    for epoch_ix in range(configs['epochs']):
        print('Starting epoch {}'.format(epoch_ix+1), flush=True)
        for i in range(num_epoch_steps):
            iters += 1
            instances = train.next_items(configs['batch_size'])

            # Compute iter_loss
            iter_loss = reranker(instances, train_ontology, is_training=True)[0]
            iter_loss = iter_loss / gradient_accumulation_steps
            iter_loss.backward()
            batch_loss += iter_loss.data.item()

            # Update params
            if iters % gradient_accumulation_steps == 0:
                accumulated_loss.update(batch_loss)
                torch.nn.utils.clip_grad_norm_(reranker.parameters(), configs['max_grad_norm'])
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = 0

            # Report loss
            if iters % configs['report_frequency'] == 0:
                print('{} Average Loss = {}'.format(iters, round(accumulated_loss(), 3)), flush=True)
                accumulated_loss = RunningAverage()

        if (epoch_ix + 1) % configs['epoch_evaluation_frequency'] > 0: continue

        # Evaluation after each epoch
        with torch.no_grad():
            start_dev_eval_time = time.time()
            print('Evaluation on the dev set')
            dev_results = evaluate(reranker, dev, dev_ontology, configs)
            print(dev_results)
            dev_score = dev_results['top1_accuracy']
            print('Evaluation on the dev set took %s seconds'  % (time.time() - start_dev_eval_time))

        # Save the reranker if it has better dev score
        if dev_score > best_dev_score:
            best_dev_score = dev_score
            print('Evaluation on the test set')
            test_results = evaluate(reranker, test, test_ontology, configs)
            print(test_results)
            final_test_results = test_results
            # Save the reranker
            save_path = join(configs['save_dir'], 'model.pt')
            torch.save({'model_state_dict': reranker.state_dict()}, save_path)
            print('Saved the reranker', flush=True)

    print(final_test_results)
    return final_test_results

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--cg_config', default='cg_basic')
    parser.add_argument('--reranker_config', default='reranker_basic')
    parser.add_argument('--dataset', default=COMETA, choices=DATASETS)
    parser.add_argument('--cg_trained_path', default=None)
    args = parser.parse_args()
    if args.cg_trained_path is None:
        args.cg_trained_path = f'/shared/nas/data/m1/tuanml/biolinking/trained_models/{args.dataset}/{args.cg_config}/model.pt'

    # Prepare configs
    cg_configs = prepare_configs(args.cg_config, args.dataset, verbose=False)              # Candidate Generator
    cg_configs['trained_path'] = args.cg_trained_path
    reranker_configs = prepare_configs(args.reranker_config, args.dataset, verbose=False)  # Reranker

    # Train the reranker
    train_reranker(cg_configs, reranker_configs)
