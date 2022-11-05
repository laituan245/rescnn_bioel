import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import math
import time
import random
import numpy as np

from transformers import *
from math import ceil, floor
from models.base import *
from os.path import dirname, join, realpath, isfile
from constants import *
from models.modules import *
from models.helpers import *
from os.path import join
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner

class BaseLightWeightModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model_type = CANDIDATES_GENERATOR

        # Base Embeddings
        self.embedding = BertEmbeddings(BertConfig.from_json_file(configs['embedding_configs_fp']))
        embedding_saved_path = configs['embedding_saved_path']
        if isfile(embedding_saved_path):
            ckpt = torch.load(embedding_saved_path, map_location=self.device)
            self.embedding.load_state_dict(ckpt['model_state_dict'], strict=False)
            for p in self.embedding.parameters(): p.requires_grad = False # Freeze
            print(f'Reloaded embeddings from {embedding_saved_path}')

        # Loss Function and Miner
        self.loss_fct = MultiSimilarityLoss(alpha=configs['loss_scale_pos'],
                                            beta=configs['loss_scale_neg'],
                                            base=configs['loss_thresh'])
        self.miner_fct = MultiSimilarityMiner(configs['loss_lambda'])

    def forward(self, instances, ontology, is_training, return_encoding_time=False):
        self.train() if is_training else self.eval()
        configs = self.configs
        nb_instances = len(instances)
        if not is_training:
            assert(not ontology.namevecs_index is None)

        # Encode instances' mentions (First Encoder)
        instance_inputs = [inst.mention['term'] for inst in instances]
        mentions_reps, _, encoding_time = \
            self.encode_texts(instance_inputs, return_encoding_time=True)

        # Training or Inference
        if is_training:
            # All entities in the minibatch are candidates
            candidates_eids, candidates_texts = self.generate_candidates(instances, ontology)
            candidates_reps = self.encode_texts(candidates_texts)[0]

            # Compute Loss
            all_mentions_eids = list(set(candidates_eids))
            candidate_labels = [all_mentions_eids.index(eid) for eid in candidates_eids]
            candidate_labels = torch.LongTensor(candidate_labels).to(self.device)
            # mention_labels
            mention_labels = [all_mentions_eids.index(inst.mention['entity_id']) for inst in instances]
            mention_labels = torch.LongTensor(mention_labels).to(self.device)
            # all_reps and all_labels
            all_reps = torch.cat([mentions_reps, candidates_reps], dim=0)
            all_labels = torch.cat([mention_labels, candidate_labels], dim=0)
            assert(all_reps.size()[0] == all_labels.size()[0])
            # compute loss
            miner_output = self.miner_fct(all_reps, all_labels)
            loss = self.loss_fct(all_reps, all_labels, miner_output)
            return loss, [], [], []
        else:
            # Infer the closest entities by querying the ontology's index
            preds, candidate_names, candidate_dists = [], [], []
            candidates_eids = ontology.all_names_eids
            for i in range(nb_instances):
                cur_preds = []
                v = mentions_reps[i, :].squeeze()
                v = F.normalize(v, dim=0, p=2).cpu().data.numpy()
                nns_indexes, nns_distances = ontology.namevecs_index.search_batched(np.reshape(v, (1, -1)))
                nns_indexes = nns_indexes.squeeze().tolist()
                nns_distances = nns_distances.squeeze().tolist()
                # Update cur_candidate_names and cur_candidate_dists
                cur_candidate_names = [ontology.name_list[index] for index in nns_indexes]
                cur_candidate_dists = nns_distances
                # for z in range(len(cur_candidate_names)-1, 0, -1):
                #     if cur_candidate_names[z].entity_id > cur_candidate_names[z-1].entity_id \
                #     and cur_candidate_dists[z] == cur_candidate_dists[z-1]:
                #         cur_candidate_names[z], cur_candidate_names[z-1] = cur_candidate_names[z-1], cur_candidate_names[z]
                # Update candidate_names and candidate_dists
                candidate_names.append(cur_candidate_names)
                candidate_dists.append(cur_candidate_dists)
                # Update preds
                for index in nns_indexes:
                    _eid = candidates_eids[index]
                    if not _eid in cur_preds:
                        cur_preds.append(_eid)
                    if len(cur_preds) == 20: break
                #assert(len(cur_preds) == 20)
                preds.append(cur_preds)
            # Sanity checks
            assert(len(candidate_names) == len(candidate_dists))
            assert(len(candidate_names[0]) == len(candidate_dists[0]))

            if return_encoding_time:
                return 0, preds, candidate_names, candidate_dists, encoding_time
            return 0, preds, candidate_names, candidate_dists

    def generate_candidates(self, instances, ontology):
        if self.model_type == PRETRAINING_MODEL:
            candidates_eids = [inst.selected_positive.entity_id for inst in instances]
            candidates_texts = [inst.selected_positive.name_str for inst in instances]
        else:
            # All entities in the minibatch are candidates
            configs = self.configs
            candidates_eids = [inst.mention['entity_id'] for inst in instances]
            if configs['hard_negatives_training']:
                for inst in instances:
                    added = 0
                    for ent in inst.candidate_entities:
                        if added == configs['max_hard_candidates']: break
                        if ent == inst.mention['entity_id']: continue
                        candidates_eids.append(ent)
                        added += 1
            candidates_texts = [random.choice(ontology.eid2names[eid]) for eid in candidates_eids]
        return candidates_eids, candidates_texts

class LightWeightModel(BaseLightWeightModel):
    def __init__(self, configs):
        BaseLightWeightModel.__init__(self, configs)

        self.tokenizer = AutoTokenizer.from_pretrained(configs['tokenizer'], use_fast=True)

        if configs['cnn_type'] == 'vdcnn':
            print('Use VCDNN')
            self.cnn_encoder = VDCNN(embedding_dim=768, depth=configs['vdcnn_cnn_depth'],
                                     n_fc_neurons=configs['feature_size'],
                                     kernel_size=configs['vdcnn_cnn_kernel_size'],
                                     padding=configs['vdcnn_cnn_padding'],
                                     dropout=configs['vdcnn_dropout'])
        elif configs['cnn_type'] == 'cnn_text':
            print('use CNN_Text')
            self.cnn_encoder = CNN_Text(input_dim=768,
                                        pooling_type=configs['pooling_type'],
                                        hidden_dim=configs['feature_size'],
                                        depth=configs['cnn_text_depth'],
                                        dropout=configs['cnn_text_dropout'])
        else:
            raise NotImplementedError

        # Move to device
        self.to(self.device)

    def encode_texts(
            self, texts, return_encoding_time=False,
            output_hidden_states=False,
        ):
        assert(not output_hidden_states)
        max_length = self.configs['max_length']
        toks = self.tokenizer.batch_encode_plus(texts,
                                                padding=True,
                                                return_tensors='pt',
                                                truncation=True,
                                                max_length=max_length,
                                                add_special_tokens=False)
        toks = toks.to(self.device)
        start_encoding_time = time.time()
        embeds = self.embedding(input_ids=toks['input_ids'],
                                token_type_ids=toks['token_type_ids'])
        embeds = embeds * toks['attention_mask'].unsqueeze(-1)
        outputs = self.cnn_encoder(embeds, toks['attention_mask'])
        encoding_time = time.time() - start_encoding_time

        if return_encoding_time:
            return outputs, [], encoding_time
        return outputs, []
