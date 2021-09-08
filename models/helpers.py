import os
import torch
import numpy as np
import json
import random

from operator import itemgetter
from os.path import exists, join
from transformers.tokenization_utils_base import BatchEncoding

def flatten(l):
    return [item for sublist in l for item in sublist]

def list_rindex(li, x):
    for i in reversed(range(len(li))):
        if li[i] == x:
            return i

def tolist(torch_tensor):
    return torch_tensor.cpu().data.numpy().tolist()

def shuffle_input_ids(toks, ngram=1):
    input_ids, attention_mask = toks['input_ids'], toks['attention_mask']
    input_ids, attention_mask = tolist(input_ids), tolist(attention_mask)
    shuffle_input_ids = []
    for _input_id, _attention_mask in zip(input_ids, attention_mask):
        end_idx = list_rindex(_attention_mask, 1)+1
        sliced_input_id = _input_id[1:end_idx-1]
        assert(len(sliced_input_id) == _attention_mask.count(1)-2)
        # Shuffling
        _ngrams = []
        for i in range(0, len(sliced_input_id), ngram):
            _ngrams.append(sliced_input_id[i:i+ngram])
        random.shuffle(_ngrams)
        sliced_input_id = flatten(_ngrams)
        # Update shuffle_input_ids
        new_input_ids = [_input_id[0]] + sliced_input_id + [_input_id[end_idx-1]] + _input_id[end_idx:]
        shuffle_input_ids.append(new_input_ids)
    shuffle_input_ids = torch.LongTensor(shuffle_input_ids)
    return shuffle_input_ids

class AllNamesCache:
    def __init__(self, base_dir):
        if base_dir is None:
            self.initialized = False
            return
        names_arrays_fp = join(base_dir, 'umls_names.npy')
        names_strs_fp = join(base_dir, 'umls_names.txt')
        if exists(names_arrays_fp) and exists(names_strs_fp):
            print('Input cache files exist')
            with open(names_arrays_fp, 'rb') as f:
                self.names_input_ids = np.load(f)
                self.names_token_type_ids = np.load(f)
                self.names_attention_mask = np.load(f)
            print(f'Cache input_ids: {self.names_input_ids.shape}')
            print(f'Cache token_type_ids: {self.names_token_type_ids.shape}')
            print(f'Cache attention_mask: {self.names_attention_mask.shape}')
            with open(names_strs_fp, 'r') as f:
                names_strs, name2index = [], {}
                for line in f:
                    cur_name_str = line.strip()
                    names_strs.append(cur_name_str)
                    name2index[cur_name_str] = len(names_strs) - 1
                self.names_strs = names_strs
                self.name2index = name2index
                # Sanity checks
                for n in self.names_strs:
                    assert(n == self.names_strs[self.name2index[n]])
                print('Passed sanity checks')
            self.initialized = True
        else:
            self.initialized = False

    def encode(self, texts):
        indices = itemgetter(*texts)(self.name2index)
        cur_input_ids = self.names_input_ids[indices, :]
        cur_token_type_ids = self.names_token_type_ids[indices, :]
        cur_attention_masks = self.names_attention_mask[indices, :]
        batch = {
            'input_ids': torch.LongTensor(cur_input_ids),
            'token_type_ids': torch.LongTensor(cur_token_type_ids),
            'attention_mask': torch.LongTensor(cur_attention_masks),
        }
        return BatchEncoding(batch)
