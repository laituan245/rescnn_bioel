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
from tqdm import tqdm
from data import load_data
from data.base import *
from scorer import evaluate
from argparse import ArgumentParser
from os.path import join
from sklearn.cluster import KMeans, MiniBatchKMeans

def load_and_benchmark_pretrained_model(config_name, saved_path, datasets = DATASETS):
    configs = prepare_configs(config_name, None)
    if 'cg' in config_name:
        if not configs['online_kd']:
            print('class DualBertEncodersModel')
            model = DualBertEncodersModel(configs)
        else:
            print('class EncodersModelWithOnlineKD')
            model = EncodersModelWithOnlineKD(configs)
    elif 'dummy' in config_name:
        assert(len(datasets) == 1)
        configs['dataset'] = datasets[0]
        model = DummyModel(configs)
    else:
        raise NotImplementedError
    if (not saved_path is None) and os.path.exists(saved_path):
        checkpoint = torch.load(saved_path, map_location=model.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('Reloaded from ckpt')
    print(f'Prepared model ({get_n_params(model)} params)')
    # Benchmarking
    weighted_avg = benchmark_model(model, configs['batch_size'], datasets)

    return weighted_avg

def benchmark_model(model, batch_size, datasets = DATASETS, split = 'test'):
    assert(model.model_type in [CANDIDATES_GENERATOR, DUMMY_MODEL, PRETRAINING_MODEL])
    configs = {'batch_size': batch_size}

    # Main Loop
    all_results, total_ctx = [], 0
    for dataset in datasets:
        assert(dataset in DATASETS)
        print(f'\nEvaluating on {dataset}')
        configs['dataset'] = dataset
        _, dev, test, ontology = load_data(dataset)
        insts = dev if split == 'dev' else test
        print('Building the ontology')
        if not model.model_type == DUMMY_MODEL:
            ontology.build_index(model, 256)
        eval_results = evaluate(model, insts, ontology, configs)
        print(f'Evaluation results on {split} of {dataset}: {eval_results}')
        all_results.append((eval_results['top1_accuracy'], len(insts)))
        total_ctx += len(insts)

    # Compute weighted avg score
    weighted_avg = 0.0
    for acc, ctx in all_results:
        weighted_avg += (acc * ctx / total_ctx)

    return weighted_avg

def generate_synthetic_data_using_bart(
        transfomer = 'facebook/bart-base', dataset = 'ncbi-disease',
        batch_size = 128, max_length = 25, num_return_sequences = 3
    ):
    assert(dataset in DATASETS)

    output_fp = join(BASE_SYNTHETIC_DATA_PATH, 'full_{}.txt'.format(dataset))

    # Prepare tokenizer, model, and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(transfomer)
    tokenizer = AutoTokenizer.from_pretrained(transfomer, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(transfomer, config=config)
    model.to(device)
    if dataset in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset}_train.json'))
    else:
        ontology = load_data(dataset)[-1]
    print('Prepared tokenizer, model, and ontology')

    # Generate a synonym for each name in the ontology
    eid2generated = {}
    all_entities = AugmentedList(ontology.entity_list)
    num_iters = math.ceil(len(all_entities) / batch_size)
    for iter in tqdm(range(num_iters)):
        batch_entities = all_entities.next_items(batch_size)
        batch = tokenizer(
            [e.primary_name.name_str for e in batch_entities],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            generated_ids = model.generate(batch['input_ids'].to(device), num_beams=5, num_return_sequences=num_return_sequences)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for ix, e in enumerate(batch_entities):
            if not e.entity_id in eid2generated: eid2generated[e.entity_id] = set()
            existing_texts = set([n.name_str.lower() for n in e.names])
            # Add generated texts
            start_ix = ix * num_return_sequences
            end_ix = (ix + 1) * num_return_sequences
            for gen in generated_texts[start_ix:end_ix]:
                if gen.lower() in existing_texts: continue
                eid2generated[e.entity_id].add(gen.lower())

    # Output synthetic training pairs
    total_nb_examples = 0
    with open(output_fp, 'w+') as output_f:
        for eid in eid2generated:
            e_name_strs = list(eid2generated[eid])
            for name_str in e_name_strs:
                assert(not '\t' in name_str)
                output_f.write(f'{eid}\t{name_str}\n')
                total_nb_examples += 1

    # Logs
    print(f'total_nb_examples: {total_nb_examples}')

def generate_pairs_from_dataset(dataset):
    output_fn = f'{dataset}_pairs.txt'
    output_fp = join('/shared/nas/data/m1/tuanml/biolinking/data/pairs_for_bart', output_fn)
    output_f = open(output_fp, 'w+')

    # Load train set and the ontology
    train, _, _, ontology = load_data(dataset)
    if dataset in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset}_train.json'))
    print('Loaded train set and the ontology')

    # Generate pairs
    total_nb_pairs = 0
    for inst in tqdm(train.items):
        entity_id, term = inst.mention['entity_id'], inst.mention['term']
        entity_names = ontology.eid2names[entity_id]
        for entity_name in entity_names:
            output_f.write(f'{entity_id}\t{entity_name}\t{term}\n')
        total_nb_pairs += 1
    print(f'total_nb_pairs: {total_nb_pairs}')

    # Close the file
    output_f.close()

def generate_pairs_from_datasets(datasets = DATASETS):
    for dataset in datasets:
        print(f'Processing {dataset}')
        generate_pairs_from_dataset(dataset)

def cluster_entity_names(
        dataset, batch_size = 256, n_clusters = 2500,
        transformer = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
        debug_mode = False
    ):
    output_fp = join(BASE_CLUSTERS_INFO_PATH, f'{dataset}_clusters.txt')

    # Load the ontology
    if dataset in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset}_train.json'))
    else:
        ontology = load_data(dataset)[-1]
    print('Loaded the ontology')

    # Load model and tokenizer
    model = AutoModel.from_pretrained(transformer)
    tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Loaded model and tokenizer')

    # Encode every entity name
    print('Encoding every entity name')
    start_time = time.time()
    namestr2vector = {}
    entity_names = AugmentedList([e.primary_name.name_str for e in ontology.entity_list])
    if debug_mode:
        entity_names.items = entity_names.items[:1000]
    nb_iters = math.ceil(len(entity_names) / batch_size)
    for _ in range(nb_iters):
        batch_names = entity_names.next_items(batch_size)
        with torch.no_grad():
            reprs = encode_texts_bert(model, tokenizer, batch_names, device)
        for ix in range(batch_size):
            namestr2vector[batch_names[ix]] = reprs[ix, :].squeeze().cpu().data.numpy()
    print('Encoding took {} seconds'.format(time.time() - start_time))
    print('Encoded every entity name')

    # K-means
    names, X = [], []
    for name in namestr2vector:
        names.append(name)
        X.append(namestr2vector[name])
    X = np.array(X)
    print('Start fitting K-means')
    start_time = time.time()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1000, random_state=0).fit(X)
    print('kmeans took {} seconds'.format(time.time() - start_time))

    # Aggregate names into clusters
    print('Start aggregating names into clusters')
    clusters = []
    for _ in range(n_clusters): clusters.append([])
    for ix, label in enumerate(kmeans.labels_):
        clusters[label].append(names[ix])

    # Output
    print('Outputting')
    with open(output_fp, 'w+') as f:
        for cluster in clusters:
            _cluster = [n.lower() for n in cluster]
            f.write(json.dumps(_cluster))
            f.write('\n')

    return clusters

def encode_texts_bert(model, tokenizer, texts, device, max_length=25):
    toks = tokenizer.batch_encode_plus(texts,
                                       padding=True,
                                       return_tensors='pt',
                                       truncation=True,
                                       max_length=max_length)
    toks = toks.to(device)
    outputs = model(**toks)
    reprs = outputs[0][:, 0, :]
    return reprs

def generate_pairs_from_clusters(dataset, nb_per_clusters=5):
    output_fp = join(BASE_SYNTHETIC_DATA_PATH, '{}.txt'.format(dataset))
    output_f = open(output_fp, 'w+')

    # Load Ontology
    pname2eid = {}
    ontology = load_data(dataset)[-1]
    for e in ontology.entity_list:
        pname2eid[e.primary_name.name_str.lower()] = e.entity_id
    print('Loaded the dataset ontology')

    # Read full generated pairs
    eid2generateds = {}
    full_pairs = join(BASE_SYNTHETIC_DATA_PATH, 'full_{}.txt'.format(dataset))
    with open(full_pairs, 'r') as f:
        for line in f:
            es = line.strip().split('\t')
            eid, generated = es
            if not eid in eid2generateds: eid2generateds[eid] = []
            eid2generateds[eid].append(generated)

    # Read Clusters
    cluster_sizes = {}
    clusters_fp = join(BASE_CLUSTERS_INFO_PATH, f'{dataset}_clusters.txt')
    with open(clusters_fp, 'r') as f:
        for line in f:
            _cluster = json.loads(line)
            selected_texts = random.sample(_cluster, min(nb_per_clusters, len(_cluster)))
            selected_texts = [t.lower() for t in selected_texts]
            selected_eids = [pname2eid[t] for t in selected_texts]
            selected_eids = [eid for eid in selected_eids if eid in eid2generateds]
            selected_generated = []
            for eid in selected_eids:
                selected_generated.append(random.choice(eid2generateds[eid]))
            # Output
            for eid, generated in zip(selected_eids, selected_generated):
                output_f.write(f'{eid}\t{generated}\n')
            # Update cluster_sizes
            c_size = len(_cluster)
            cluster_sizes[c_size] = cluster_sizes.get(c_size, 0) + 1
    #print('Cluster size')
    #print(cluster_sizes)

    output_f.close()
