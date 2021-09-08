import time
import json
import random
import requests
import numpy as np
import math
import re
from tqdm import tqdm

from utils import *
from constants import *
from transformers import *
from os.path import join
from data.base import Ontology
from elasticsearch import Elasticsearch

def extract_batches(all_lines, batch_size, direction='forward'):
    assert(direction in ['forward', 'backward'])

    if direction == 'forward':
        all_lines.sort(key=lambda x: x.split('\t')[1], reverse=True)
    elif direction == 'backward':
        all_lines.sort(key=lambda x: x.split('\t')[1][::-1], reverse=True)

    # Extract batches
    all_batches, cur_idx = [], 0
    all_eids = [l.split('\t')[0] for l in all_lines]
    all_terms1 = [l.split('\t')[1] for l in all_lines]
    used = [False for l in all_lines]
    nb_batches = int(len(all_lines) / batch_size)
    for batch_idx in tqdm(range(nb_batches)):
        cur_batch, cur_appeared = [], set()
        while cur_idx < len(all_lines) and used[cur_idx]:
            cur_idx += 1
        for i in range(cur_idx, len(all_lines)):
            if used[i]: continue
            if not all_terms1[i] in cur_appeared:
                cur_batch.append(all_lines[i])
                cur_appeared.add(all_terms1[i])
                used[i] = True
            if len(cur_batch) == batch_size:
                break
        all_batches.append(cur_batch)

    # Remaining batch
    remaining_batch = []
    for i in range(len(all_lines)):
        if not used[i]:
            remaining_batch.append(all_lines[i])
    all_batches.append(remaining_batch)

    random.shuffle(all_batches)
    return all_batches

def sort_pretrain_pairs(input_fp='pretrain_positive_pairs.txt',
                        output_fp='sorted_pretrain_positive_pairs.txt',
                        batch_size = 8, direction='all'):
    assert(direction in ['forward', 'backward', 'all'])
    all_lines = []
    with open(input_fp, 'r', encoding='utf-8') as f:
        for line in f:
            all_lines.append(line.strip())
    print(f'Total number of lines: {len(all_lines)}')

    all_batches = []
    if direction in ['forward', 'all']:
        forward_batches = extract_batches(all_lines, batch_size, direction='forward')
        all_batches += forward_batches
    if direction in ['backward', 'all']:
        backward_batches = extract_batches(all_lines, batch_size, direction='backward')
        all_batches += backward_batches
    random.shuffle(all_batches)
    batched_lines = flatten(all_batches)

    # Output file
    with open(output_fp, 'w+', encoding='utf-8') as output_f:
        for line in batched_lines:
            output_f.write(f'{line.strip()}\n')


def generate_pretrain_pairs(
        output_fp='pretrain_positive_pairs.txt',
        ignore_trivial_pairs=False,
        cc_fp=None):
    # Read cc file
    eid2cc = {}
    if not cc_fp is None:
        with open(cc_fp, 'r') as f:
            for ix, line in enumerate(f):
                cc = eval(line)
                for eid in cc: eid2cc[eid] = f'synset_{ix}'
        print('Read cc file')

    # Prepare output file
    output_f = open(output_fp, 'w+', encoding='utf-8')

    # Load ontology
    ontology = Ontology(UMLS_2020AA_FULL_FP, eid2cc)

    # Generate
    total_nb_pairs, total_nb_entities, trivial_pairs_ignored = 0, 0, 0
    for e in tqdm(ontology.entity_list):
        e_name_strs = list(set([n.name_str.lower().strip() for n in e.names]))
        if len(e_name_strs) == 1: continue
        # Generate pairs
        pairs = []
        for n1 in range(len(e_name_strs)):
            for n2 in range(n1+1, len(e_name_strs)):
                pairs.append((e_name_strs[n1], e_name_strs[n2]))
        random.shuffle(pairs)
        # Ignore trivial pairs (if enabled)
        if ignore_trivial_pairs:
            filtered = []
            for n1, n2 in pairs:
                n1_cleaned = re.sub(r'[^a-zA-Z0-9]', '', n1).lower().strip()
                n2_cleaned = re.sub(r'[^a-zA-Z0-9]', '', n2).lower().strip()
                if n1_cleaned == n2_cleaned:
                    trivial_pairs_ignored += 1
                    continue
                filtered.append((n1, n2))
            pairs = filtered
        # Select at most 50 pairs
        pairs = pairs[:50]
        # Write to file
        total_nb_entities += 1
        for pair in pairs:
            name1, name2 = pair[0], pair[1]
            assert(not '\t' in name1)
            assert(not '\t' in name2)
            output_f.write(f'{e.entity_id}\t{name1}\t{name2}\n')
            total_nb_pairs += 1
    # Close the output file
    print(f'total_nb_pairs {total_nb_pairs}')
    print(f'total_nb_entities {total_nb_entities}')
    print(f'trivial_pairs_ignored {trivial_pairs_ignored}')
    output_f.close()


def generate_hard_negatives(output_fp='hard_negatives.txt', topk=25):
    output_file = open(output_fp, 'w+', encoding='utf-8')

    # Read UMLS ontology
    ontology = Ontology(UMLS_2017AA_ACTIVE_FP)

    # Compute number of iters
    batch_size = 100
    entity_list = AugmentedList(ontology.entity_list)
    all_iters = int(math.ceil(len(entity_list) / batch_size))

    # Generate negative pairs
    for iter in tqdm(range(all_iters)):
        batch_entities = entity_list.next_items(batch_size)
        batch_names_strs = [e.primary_name.name_str for e in batch_entities]
        batch_entity_ids = [e.entity_id for e in batch_entities]
        hard_negatives = search_hard_negatives(batch_names_strs, batch_entity_ids, size=topk)
        for i, e in enumerate(batch_entities):
            row = [e.entity_id] + hard_negatives[i]
            row_str = json.dumps(row)
            output_file.write(f'{row}\n')

    # Close the output file
    output_file.close()

def index_names(index='umls', doc_type = 'entity_name'):
    # Read UMLS ontology
    ontology = Ontology(UMLS_2017AA_ACTIVE_FP)

    # ElasticSearch
    es = Elasticsearch()
    es.indices.delete(index=index, ignore=[400, 404]) # Delete the existing index (if exists)
    response = es.indices.create(index=index, ignore=400)        # Create new index

    # Index each name into elasticsearch
    ctx = 0
    start_time = time.time()
    for n in ontology.name_list:
        new_name = {
            'name_string': n.name_str,
            'entity_id': n.entity_id
        }
        response = es.index(
            index = index,
            doc_type = doc_type,
            body = new_name
        )
        ctx += 1
        if ctx % 10000 == 0:
            elapsed_time = time.time() - start_time
            print(f'Indexed {ctx} ({elapsed_time} seconds)')
    es.indices.refresh(index)
    docs_count = es.cat.count(index, params={"format": "json"})
    print(f'Number of indexes names: {docs_count}')

# Bulk search
def search_hard_negatives(query_names, query_entity_ids, index='umls', size=25):
    assert(len(query_names) == len(query_entity_ids))

    # ElasticSearch
    es = Elasticsearch()
    hard_negatives = []

    # Create query_data
    queries = []
    for query_name in query_names:
        query = {
            'size': size,
            'query': {
                'match': {
                    'name_string': query_name
                }
            }
        }
        queries.append({"index": index, "type": "entity_name"})
        queries.append(query)
    responses = es.msearch(body=queries)['responses']
    for ix, response in enumerate(responses):
        data = response['hits']['hits']
        cur_hard_negatives = []
        cur_query_name = query_names[ix]
        cur_query_entity_id = query_entity_ids[ix]
        for hit in data:
            s = hit['_source']
            if s['entity_id'] == cur_query_entity_id: continue
            if s['name_string'].lower() == cur_query_name.lower(): continue
            cur_hard_negatives.append(s['entity_id'].lower())
        cur_hard_negatives = list(set(cur_hard_negatives))
        hard_negatives.append(cur_hard_negatives)

    return hard_negatives

def tokenize_umls_names(transformer='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                        output_dir='/shared/nas/data/m1/tuanml/umls_pretrain_data/',
                        max_length=24):
    # Read UMLS ontology
    ontology = Ontology(UMLS_2017AA_ACTIVE_FP)
    names = [n.name_str for n in ontology.name_list]

    # Tokenization
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(transformer, use_fast=True)
    toks = tokenizer.batch_encode_plus(names,
                                       padding=True,
                                       return_tensors='pt',
                                       truncation=True,
                                       max_length=max_length)
    print(f'Total Tokenization Time: {time.time() - start_time} secs')
    input_ids = toks['input_ids'].numpy()
    token_type_ids = toks['token_type_ids'].numpy()
    attention_mask = toks['attention_mask'].numpy()

    # Save to files
    print('Saving to files')
    with open(join(output_dir, 'umls_names.npy'), 'wb') as f:
        np.save(f, input_ids)
        np.save(f, token_type_ids)
        np.save(f, attention_mask)
    with open(join(output_dir, 'umls_names.txt'), 'w+') as f:
        for n in names:
            f.write(f'{n}\n')
    print('Saved')
