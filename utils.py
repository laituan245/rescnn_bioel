import os
import math
import torch
import pickle
import pyhocon
import random
import time
import numpy as np

from tqdm import tqdm
from os.path import join
from models import *
from constants import *
from transformers import *
from unidecode import unidecode

def apply_model(model, instances, ontology, batch_size, disable_tqdm = False):
    model_inference_time = 0
    inference_iters = math.ceil(len(instances) / batch_size)
    for iter in tqdm(range(inference_iters), disable=disable_tqdm):
        batch_instances = instances.next_items(batch_size)
        with torch.no_grad():
            outputs = model(batch_instances, ontology, is_training=False, return_encoding_time=True)
            candidate_entities, candidate_names, candidate_distances, encoding_time = outputs[1:]
            model_inference_time += encoding_time
        for ix, inst in enumerate(batch_instances):
            inst.candidate_entities = candidate_entities[ix]
            inst.candidate_names = candidate_names[ix]
            inst.candidate_distances = candidate_distances[ix]
            if model.model_type == CANDIDATES_GENERATOR:
                inst.should_be_reranked = True # default
    return model_inference_time

def replace_non_ascii(text):
    ver1 = unidecode(text)
    ver2 = ''.join([i if ord(i) < 128 else '_' for i in text])
    if len(ver1) == len(text): return ver1
    return ver2

def prepare_configs(
    config_name,
    dataset,
    verbose=True,
    transformer=None
):
    if config_name is None: return None

    # Extract the requested config
    if verbose: print('Config {}'.format(config_name), flush=True)
    configs = pyhocon.ConfigFactory.parse_file(BASIC_CONF_PATH)[config_name]
    if dataset: configs['dataset'] = dataset

    # save_dir
    if dataset:
        configs['save_dir'] = join(join(BASE_SAVE_PATH, configs['dataset']), config_name)
        create_dir_if_not_exist(configs['save_dir'])

    #
    if not transformer is None:
        configs['transformer'] = transformer

    if verbose: print(configs, flush=True)
    return configs

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_n_tunable_params(model):
    pp=0
    for p in list(model.parameters()):
        if not p.requires_grad: continue
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def create_dir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def tolist(torch_tensor):
    return torch_tensor.cpu().data.numpy().tolist()

def flatten(l):
    return [item for sublist in l for item in sublist]

def listRightIndex(alist, value):
    return len(alist) - alist[-1::-1].index(value) -1

def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))

def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)

def initialize_bert_student(teacher_transformer, student_config_fp, save_dir):
    # Intialize the teacher
    teacher = AutoModel.from_pretrained(teacher_transformer)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_transformer, use_fast=True)
    teacher_dict = teacher.state_dict()
    assert('BertModel' in teacher.config.architectures)
    print('Intialized the teacher ({} params)'.format(get_n_params(teacher)))

    # Initialize the student
    student_config = BertConfig.from_json_file(student_config_fp)
    student = BertModel(student_config)
    student_dict = student.state_dict()
    print('Intialized the student ({} params)'.format(get_n_params(student)))

    # Load param weights of the teacher into the student
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in teacher_dict.items() if k in student_dict}
    #print(pretrained_dict.keys())
    # 2. overwrite entries in the existing state dict
    student_dict.update(pretrained_dict)
    # 3. load the new state dict
    student.load_state_dict(pretrained_dict)

    # Save the student model
    create_dir_if_not_exist(save_dir)
    student.save_pretrained(save_dir)

    # Save the tokenizer
    teacher_tokenizer.save_pretrained(save_dir)

def save_bert_embeddings(transformer='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                         save_dir='/shared/nas/data/m1/tuanml/biolinking/initial_embeddings/sapbert/'):
    create_dir_if_not_exist(save_dir)
    # Load model and embeddings
    model = AutoModel.from_pretrained(transformer)
    embeddings = model.embeddings
    print(f'type(embeddings): {type(embeddings)}')
    print(f'Nb Params: {get_n_params(embeddings)}')
    # Save
    save_path = join(save_dir, 'embedding.pt')
    torch.save({'model_state_dict': embeddings.state_dict()}, save_path)
    print('Saved the model', flush=True)

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        if self.steps == 0: return 'NA'
        return self.total/float(self.steps)

class AugmentedList:
    def __init__(self, items, shuffle_between_epoch=False):
        self.items = items
        self.cur_idx = 0
        self.shuffle_between_epoch = shuffle_between_epoch

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            returned_batch = [item for item in first_part + second_part]
            if self.shuffle_between_epoch:
                random.shuffle(self.items)
            return returned_batch

    def __len__(self):
        return len(self.items)

    @property
    def size(self):
        return len(self.items)

# Algorithms Implementation
def KMPSearch(txt, pat):
    indexes = []
    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest prefix suffix
    # values for pattern
    lps = [0]*M
    j = 0 # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)

    i = 0 # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            indexes.append(i-j)
            j = lps[j-1]

        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j-1]
            else:
                i += 1

    return indexes

def computeLPSArray(pat, M, lps):
    len = 0 # length of the previous longest prefix suffix

    lps[0] # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i]== pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len-1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1
