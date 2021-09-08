import os
import copy
import utils
import torch
import json
import random
import math
import pyhocon
import warnings
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


from utils import *
from constants import *
from models import *
from transformers import *
from data import load_data
from data.base import *
from scorer import evaluate
from os.path import join
from argparse import ArgumentParser
from scipy.special import softmax

TRAINED_BASE_DIR = '/shared/nas/data/m1/tuanml/biolinking/trained_models/'

def load_model(config_name, dataset):
    save_path = join(join(join(TRAINED_BASE_DIR, dataset), config_name), 'model.pt')
    configs = prepare_configs(config_name, dataset, verbose=False)
    if 'lightweight' in config_name:
        model = LightWeightModel(configs)
    else:
        model = DualBertEncodersModel(configs)
    ckpt = torch.load(save_path, map_location=model.device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f'Reloaded the model from {save_path}')
    return model

def check_label(gt_entity_ids, topk_entity_ids):
    crtx_top1, crtx_top5, crtx_top10, crtx_top20 = 0, 0, 0, 0
    for gt_entity_id in gt_entity_ids:
        for i in range(min(20, len(topk_entity_ids))):
            assert(not '+' in topk_entity_ids[i])
            pred_entity_ids = topk_entity_ids[i].split('|')
            if gt_entity_id in pred_entity_ids:
                if i < 1: crtx_top1 = 1
                if i < 5: crtx_top5 = 1
                if i < 10: crtx_top10 = 1
                if i < 20: crtx_top20 = 1
    return crtx_top1, crtx_top5, crtx_top10, crtx_top20

def get_correct_cases(model, instances, ontology):
    with torch.no_grad():
        evaluation_results, correct_cases = \
            evaluate(model, instances, ontology, model.configs, True)
        print(evaluation_results)
    return correct_cases

def config_to_name(config):
    if config == 'cg_basic': return '12-layer SapBERT'
    if 'layers' in config:
        nb_layer = config.split('_')[2]
        return f'{nb_layer}-layer SapBERT'

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=COMETA, choices=DATASETS)
    parser.add_argument('--config_1', default='cg_basic')
    parser.add_argument('--config_2', default='lightweight_cnn_text')
    args = parser.parse_args()

    # Load dataset
    train, dev, test, ontology = load_data(args.dataset)
    instances = dev
    if args.dataset in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{args.dataset}_dev.json'))
    print('Loaded the dataset')

    # Load first model
    print('Loading the first model')
    first_model = load_model(args.config_1, args.dataset)

    # Load second model
    print('Loading the second model')
    second_model = load_model(args.config_2, args.dataset)

    # Applt the two models
    print('Applying the two models')
    ontology.build_index(first_model, 256)
    first_corrects = get_correct_cases(first_model, instances, ontology)
    ontology.build_index(second_model, 256)
    second_corrects = get_correct_cases(second_model, instances, ontology)

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    model1_name = config_to_name(args.config_1)
    model2_name = config_to_name(args.config_2)
    labels = [f'Only {model1_name} is correct',
              'Both are correct',
              f'Only {model2_name} is correct',
              'Both are wrong']
    sizes = [len(first_corrects - second_corrects),
             len(first_corrects.intersection(second_corrects)),
             len(second_corrects - first_corrects),
             len(instances) - len(first_corrects.union(second_corrects))]
    percentages = [round(100 * s / len(instances), 2) for s in sizes]
    percentages = [f'{p}%' for p in percentages]
    plt.pie(sizes, labels=percentages, startangle=90)
    plt.legend(labels, bbox_to_anchor=(0.85,1.025), loc='upper left')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig(f'figures/compare_{args.config_1}_{args.config_2}_{args.dataset}.png', bbox_inches='tight')
