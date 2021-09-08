import os
import copy
import utils
import torch
import json
import random
import time
import math
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
from data.base import Ontology
from scorer import evaluate
from os.path import isfile, join
from argparse import ArgumentParser
from tqdm import tqdm

TRAINED_BASE_DIR = '/shared/nas/data/m1/tuanml/biolinking/trained_models/'

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=COMETA, choices=DATASETS)
    parser.add_argument('--cg_config', default='cg_basic')
    parser.add_argument('--cg_trained_model', default=None)
    args = parser.parse_args()
    dataset_name = args.dataset

    # Train models Paths
    if args.cg_trained_model is None:
        args.cg_trained_model = join(join(join(TRAINED_BASE_DIR, args.dataset), args.cg_config), 'model.pt')

    # Load dataset
    print('')
    print(f'dataset: {args.dataset}')
    train, dev, test, ontology = load_data(args.dataset)
    if dataset_name in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset_name}_dev.json'))
    print('Loaded the dataset')

    # Prepare configs
    cg_configs = prepare_configs(args.cg_config, args.dataset, verbose=False)
    print('Prepared configs')

    # Prepare a FullModel
    model = FullModel(cg_configs)
    if isfile(args.cg_trained_model):
        print(f'Reloading {args.cg_trained_model}')
        model.reload_cg(args.cg_trained_model)
    print('Prepare a FullModel ({} Params)'.format(get_n_params(model)))

    # Build the ontology's index
    ontology_start = time.time()
    model.build_ontology_index(ontology)
    print('Built the ontology index')
    ontology_time = time.time() - ontology_start

    # Evaluations
    print('Evaluations')
    # Evaluate the candidate generator
    evaluation_start = time.time()
    cg_dev_results = evaluate(model.cg, dev, ontology, cg_configs)
    print('Candidate Generator: {}'.format(cg_dev_results))
    evaluation_time = time.time()-evaluation_start

    #
    print(f'ontology_time: {ontology_time}')
    print(f'evaluation_time: {evaluation_time}')
    print(f'total_time: {ontology_time + evaluation_time}')
