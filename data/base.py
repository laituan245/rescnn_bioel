import os
import math
import json
import time
import torch
import pickle
import pyhocon
import numpy as np
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from constants import *

try:
    import scann
except ImportError as error:
    print('scann is required')

CONTEXT_SIZE = 8

# Ontology
class OntologyNameEntry:
    def __init__(self, name_str, name_type, entity_id):
        assert(name_type in NAMETYPES)
        self.name_str = name_str
        self.name_type = name_type   # primary or secondary
        self.entity_id = entity_id   # id of the entity that the name is associated with

    def __str__(self):
        entity_id, name = self.entity_id, self.name_str
        str_repr = 'Entity ID = {} | String =  {}'.format(entity_id, name)
        return str_repr

class OntologyEntityEntry:
    def __init__(self, entity_id, names, primary_name):
        assert(not primary_name is None)
        self.entity_id = entity_id
        self.names = names
        self.primary_name = primary_name

    def __str__(self):
        entity_id, primary_name = self.entity_id, self.primary_name
        str_repr = 'Entity ID = {} | Primary Name = "{}"'.format(entity_id, primary_name.name_str)
        if len(self.names) > 1:
            str_repr += ' | Secondary Names = ['
            for n in self.names:
                if n.name_type != 'primary':
                    str_repr += '"{}"'.format(n.name_str)
            str_repr += ']'
        return str_repr

class Ontology:
    def __init__(self, fp, eid2cc=None):
        self.ontology_name = fp[fp.rfind('/')+1:]
        # No index is available initially
        self.namevecs_index = None
        # Read ontology data from fp
        with open(fp, 'r') as f:
            data = json.loads(f.read())
        # Merge entity ids in same connected component
        if not eid2cc is None:
            nb_entities_decreased = 0
            for eid in eid2cc:
                if not eid in data: continue
                cc = eid2cc[eid]
                if not cc in data:
                    data[cc] = []
                    nb_entities_decreased -= 1
                for n in data[eid]: data[cc].append(n)
                del data[eid]
                nb_entities_decreased += 1
            print(f'nb_entities_decreased: {nb_entities_decreased}')
        # Create name_list, entity_list
        name_list, entity_list = [],[]
        for entity_id in data.keys():
            entity_names, primary_name = [], None
            for n in data[entity_id]:
                name_entry = OntologyNameEntry(n[1], n[0], entity_id)
                entity_names.append(name_entry)
                name_list.append(name_entry)
                if n[0] == NAME_PRIMARY: primary_name = name_entry
            entity = OntologyEntityEntry(entity_id, entity_names, primary_name)
            entity_list.append(entity)
        self.name_list = name_list
        self.entity_list = entity_list
        # Create eid2entity
        self.eid2entity = {}
        for e in entity_list:
            self.eid2entity[e.entity_id] = e
        # Create all_entity_ids
        self.all_entity_ids = []
        for e in entity_list:
            self.all_entity_ids.append(e.entity_id)
        # Create eid2pname (entity-id -> primary-name)
        self.eid2pname = {}
        for e in entity_list:
            self.eid2pname[e.entity_id] = e.primary_name.name_str
        # Create eid2names (entity-id -> all names)
        self.eid2names = {}
        for e in entity_list:
            self.eid2names[e.entity_id] = [n.name_str for n in e.names]
        # Create all_names_eids
        self.all_names_eids = [n.entity_id for n in name_list]
        # Statistics
        nb_entities = len(self.entity_list)
        print('Statistics on the ontology')
        print('Number of entities: {}'.format(nb_entities))
        print('Number of names: {}'.format(len(self.name_list)))

    def build_index(self, model, batch_size):
        print(f'Building index for {self.ontology_name}')
        model.eval()
        start_time = time.time()
        all_vecs, ctx, total_encoding_time = [], 0, 0
        inference_iters = math.ceil(len(self.name_list) / batch_size)
        for i in tqdm(range(inference_iters)):
            batch_names = self.name_list[i*batch_size: (i+1)*batch_size]
            batch_names_strs = [n.name_str for n in batch_names]
            with torch.no_grad():
                results = model.encode_texts(batch_names_strs, return_encoding_time=True)
                batch_vectors, batch_encoding_time = results[0], results[-1]
                batch_vectors = batch_vectors.to(torch.device('cpu'))
                for j in range(batch_vectors.size()[0]):
                    x = batch_vectors[j, :].squeeze()
                    x = F.normalize(x, dim=0, p=2).cpu().data.numpy()
                    all_vecs.append(np.reshape(x, (1, -1)))
                    ctx += 1
                # Update total_encoding_time
                total_encoding_time += batch_encoding_time
        all_vecs = np.concatenate(all_vecs)
        self.namevecs_index = scann.scann_ops_pybind.builder(all_vecs, 100, 'dot_product').tree(
                num_leaves=int(math.sqrt(ctx)), num_leaves_to_search=500, training_sample_size=250000).score_ah(
                2, anisotropic_quantization_threshold=0.2).reorder(500).build()
        print('Building the index took %s seconds' % (time.time() - start_time))
        print(f'Total actual encoding time {total_encoding_time}')

# Data Instance
class DataInstance:
    def __init__(self, id, context, mention):
        # Preprocessing
        if len(context) > 0 and context[0] == '"' and context[-1] == '"':
            context = context[1:-1]

        # Update fields
        self.id = id
        self.context = context
        self.mention = mention

        self.left_context, self.right_context = '', ''
        if len(context.strip()) > 0:
            # Context words
            context_words = self.context.split(' ')
            context_words_lowered = [w.lower() for w in context_words]
            # Mention words
            mention_words = mention['term'].split(' ')
            mention_words_lowered = [w.lower() for w in mention_words]
            # Sublist search
            indexes = KMPSearch(context_words_lowered, mention_words_lowered)
            # Compute left context and right context
            try:
                assert(len(indexes) > 0)
                start_index = indexes[len(indexes) // 2]
                end_index = start_index + len(mention_words)
                assert(context_words_lowered[start_index] == mention_words_lowered[0])
                assert(context_words_lowered[end_index-1] == mention_words_lowered[-1])
                # Left Context
                left_ctx_words = context_words[max(0, start_index-CONTEXT_SIZE): start_index]
                self.left_context = ' '.join(left_ctx_words).strip()
                # Right Context
                right_ctx_words = context_words[end_index: end_index + CONTEXT_SIZE]
                self.right_context = ' '.join(right_ctx_words).strip()
            except:
                # Error occured
                pass
                #print('error occured')

class PretrainingPositivePairs:
    def __init__(self, postive_pairs_fp):
        self.positive_pairs = []
        with open(postive_pairs_fp, 'r', encoding='utf-8') as f:
            ix = 0
            for line in f:
                entity_id, name1_str, name2_str = line.strip().split('\t')
                name1 = OntologyNameEntry(name1_str, NAME_PRIMARY, entity_id)
                name2 = OntologyNameEntry(name2_str, NAME_PRIMARY, entity_id)
                self.positive_pairs.append((name1, name2))

    def __getitem__(self, item):
        return self.positive_pairs[item]
