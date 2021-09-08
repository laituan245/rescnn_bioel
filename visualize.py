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

from utils import *
from constants import *
from models import *
from transformers import *
from data import load_data
from scorer import evaluate
from os.path import join
from argparse import ArgumentParser
from scipy.special import softmax

BASE_VISUALIZATIONS_DIR = 'visualizations'
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

def eval_and_visualize(model, instances, ontology, configs, visualization_fp):
    inst2preds = {}
    with torch.no_grad():
        # Evaluation
        eval_results = evaluate(model, instances, ontology, configs)
        print(eval_results)
        # Visualization
        with open(visualization_fp, 'w+') as f:
            for inst in instances.items:
                if len(inst.context.strip()) > 0:
                    f.write('<b>Context:</b> {}<br/>'.format(inst.context))
                f.write('<b>Query Term:</b> {} </br>'.format(inst.mention['term']))

                # Info about entity
                entity = ontology.eid2entity[inst.mention['entity_id']]
                f.write('<b>Correct Entity:</b> {}</br>'.format(entity))

                # Info about candidates
                f.write('</br>')
                for ix, ceid in enumerate(inst.candidate_entities[:10]):
                    color = 'blue' if ceid == entity.entity_id else 'red'
                    f.write('<span style="color:{}">Candidate {}. {} </span></br>'.format(color, ix, ontology.eid2entity[ceid]))

                # Info about entropy
                f.write('</br>')
                f.write('<b>Entropy (Top-20 names)</b> </br>')
                candidate_distances = inst.candidate_distances[:20]
                candidate_names = inst.candidate_names[:20]
                candidate_distances = [round(d, 4) for d in candidate_distances]
                candidate_names_eids = [c.entity_id for c in candidate_names]
                # Output
                f.write(f'Distances/Similarities {candidate_distances}</br>')
                f.write(f'{candidate_names_eids}</br>')

                # Separator Line
                f.write('<hr>')
                # Update inst2preds
                inst2preds[inst.id] = inst.candidate_entities

    return inst2preds

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=COMETA, choices=DATASETS)
    parser.add_argument('--config_1', default='cg_basic')
    parser.add_argument('--config_2', default='lightweight_cnn_text')
    args = parser.parse_args()
    dataset_name = args.dataset
    base_output_dir = join(BASE_VISUALIZATIONS_DIR, args.dataset)
    create_dir_if_not_exist(base_output_dir)

    # Load dataset
    train, dev, test, ontology = load_data(dataset_name)
    print('Loaded the dataset')

    # Prepare models
    model1 = load_model(args.config_1, args.dataset)
    model2 = load_model(args.config_2, args.dataset)
    print('Loaded model1 ({} Params)'.format(get_n_params(model1)))
    print('Loaded model2 ({} Params)'.format(get_n_params(model2)))

    # Evaluate and visualize the first model
    print('First Model')
    ontology.build_index(model1, 256)
    model1_fp = join(base_output_dir, '{}.html'.format(args.config_1))
    model1_preds = eval_and_visualize(model1, test, ontology, model1.configs, model1_fp)

    # Evaluate and visualize the second model
    print('Second Model')
    ontology.build_index(model2, 256)
    model2_fp = join(base_output_dir, '{}.html'.format(args.config_2))
    model2_preds = eval_and_visualize(model2, test, ontology, model2.configs, model2_fp)

    # Visualize improvements
    improvs_output_fp = join(base_output_dir, 'improvements.html')
    with open(improvs_output_fp, 'w+') as f:
        for inst in test.items:
            inst_id = inst.id
            model1_pred = model1_preds[inst_id]
            model2_pred = model2_preds[inst_id]
            # Check if model1 is wrong but model2 is correct
            if model1_pred[0] != inst.mention['entity_id'] and model2_pred[0] == inst.mention['entity_id']:
                if len(inst.context.strip()) > 0:
                    f.write('<b>Context:</b> {}<br/>'.format(inst.context))
                f.write('<b>Query Term:</b> {} </br>'.format(inst.mention['term']))
                # Info about entity
                entity = ontology.eid2entity[inst.mention['entity_id']]
                f.write('<b>Correct Entity:</b> {}</br>'.format(entity))
                # Info about candidates (Model 1)
                f.write('</br> Model 1</br>')
                for ix, ceid in enumerate(model1_pred[:10]):
                    color = 'blue' if ceid == entity.entity_id else 'red'
                    f.write('<span style="color:{}">Candidate {}. {} </span></br>'.format(color, ix, ontology.eid2entity[ceid]))
                # Info about candidates (Model 2)
                f.write('</br> Model 2 </br>')
                for ix, ceid in enumerate(model2_pred[:10]):
                    color = 'blue' if ceid == entity.entity_id else 'red'
                    f.write('<span style="color:{}">Candidate {}. {} </span></br>'.format(color, ix, ontology.eid2entity[ceid]))
                # Separator Line
                f.write('<hr>')
