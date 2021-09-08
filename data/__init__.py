import json
import pickle
import random
import numpy as np

from utils import *
from constants import *
from os.path import join
from data.base import Ontology, DataInstance
from data.cometa import load_cometa_dataset
from data.medmentions import load_medmentions_dataset

def load_data(dataset, use_synthetic_train=False):
    assert (dataset in DATASETS)
    base_path = 'resources/{}'.format(dataset)

    print('Initializing new train, dev, test, and ontology')
    if dataset == COMETA:
        ontology = Ontology(SNOMEDCT_FP)
        train, dev, test = load_cometa_dataset(base_path, ontology)
        assert(len(ontology.entity_list) == 350830)
        assert(len(ontology.name_list) == 910823)
    if dataset == MEDMENTIONS:
        ontology = Ontology(UMLS_2017AA_ACTIVE_FP)
        train, dev, test = load_medmentions_dataset(base_path, ontology)
        assert(len(ontology.entity_list) >= 3400000)
    if dataset in [BC5CDR_C, BC5CDR_D, NCBI_D]:
        ontology = Ontology(join(BASE_ONTOLOGY_DIR, f'{dataset}.json'))
        with open(join(base_path, 'data.json'), 'r') as f:
            data = json.loads(f.read())
        traindev, train, dev, test, example_id = [], [], [], [], 0
        for split in ['traindev', 'train', 'dev', 'test']:
            for term, entity_id in data[split]:
                mention = {'term': term, 'entity_id': entity_id}
                inst = DataInstance(example_id, '', mention)
                if split == 'traindev': traindev.append(inst)
                if split == 'train': train.append(inst)
                if split == 'dev': dev.append(inst)
                if split == 'test': test.append(inst)
                example_id += 1
        if USE_TRAINDEV:
            print('use traindev')
            train = traindev

    # Use synthetic train examples (if use_synthetic_train)
    if use_synthetic_train:
        train_eids = set([inst.mention['entity_id'] for inst in train])
        print(f'Number of original train examples: {len(train)}')
        print('Loading synthetic train examples')
        synthetic_train = load_synthetic_data(join(BASE_SYNTHETIC_DATA_PATH, f'{dataset}.txt'))

        # Filtering
        print(f'Before filtering | len(synthetic_train): {len(synthetic_train)}')
        _filtered, filtered_ctx = [], 0
        for inst in synthetic_train:
            if not inst.mention['entity_id'] in train_eids:
                _filtered.append(inst)
            else:
                filtered_ctx += 1
        print(f'filtered_ctx: {filtered_ctx}')
        synthetic_train = _filtered
        print(f'After filtering | len(synthetic_train): {len(synthetic_train)}')

        # Update train
        train = train + synthetic_train

    # Convert to AugmentedLists
    random.shuffle(train)
    train = AugmentedList(train, shuffle_between_epoch=True)
    dev = AugmentedList(dev)
    test = AugmentedList(test)

    print('Train: {} examples | Dev: {} examples | Test: {} examples'.format(len(train), len(dev), len(test)))
    return train, dev, test, ontology

def load_unlabeled_data(fp):
    example_id = 0
    unlabeled_instances, all_terms, all_contexts = [], set(), set()
    with open(fp, 'r') as f:
        for line in f:
            data = json.loads(line)
            context, term = data['context'], data['entity_text']
            inst = DataInstance(example_id, context, {'term': term})
            unlabeled_instances.append(inst)
            all_contexts.add(context)
            all_terms.add(term)
            example_id += 1
    print('Number of sentences: {}'.format(len(all_contexts)))
    print('Number of unique entity terms: {}'.format(len(all_terms)))
    print('len(unlabeled_instances): {}'.format(len(unlabeled_instances)))
    return unlabeled_instances

def load_synthetic_data(fp):
    data, example_id = [], 0
    with open(fp, 'r') as f:
        for line in f:
            entity_id, term = line.strip().split('\t')
            mention = {'term': term, 'entity_id': entity_id}
            inst = DataInstance(f'synthetic_{example_id}', '', mention)
            data.append(inst)
            example_id += 1
    print(f'Number of synthetic examples: {len(data)}')
    return data
