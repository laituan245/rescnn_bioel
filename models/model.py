import json
import torch
import utils
import random
import time
import numpy as np

from utils import *
from constants import *
from models.base import *
from models.helpers import *
from os.path import join
from pytorch_metric_learning.losses import MultiSimilarityLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from transformers import AutoModel
from models.cnn import *
from transformers.adapters.configuration import ParallelConfig

# Main Classes
class DualBertEncodersModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model_type = CANDIDATES_GENERATOR
        self.all_names_cache = AllNamesCache(None)

        self.transformer = AutoModel.from_pretrained(configs['transformer'])
        task_name = 'umls-synonyms'
        adapter_config = ParallelConfig()
        self.transformer.add_adapter(task_name, config=adapter_config)
        self.transformer.train_adapter([task_name])
        self.transformer.set_active_adapters([task_name])
        if configs['gradient_checkpointing']:
            self.transformer.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], use_fast=True)

        # FFNN for dim reduction (if enabled)
        self.transformer_feature_size = self.transformer.config.hidden_size
        if configs['feature_proj']:
            self.feature_size = configs['feature_size']
            hidden_size = (self.transformer_feature_size + self.feature_size) // 2
            self.ffnn = FFNNModule(self.transformer_feature_size,
                                   [hidden_size],
                                   self.feature_size)
        else:
            self.feature_size = self.transformer_feature_size
            self.ffnn = None

        # Loss Function and Miner
        self.loss_fct = MultiSimilarityLoss(alpha=configs['loss_scale_pos'],
                                            beta=configs['loss_scale_neg'],
                                            base=configs['loss_thresh'])
        self.miner_fct = MultiSimilarityMiner(configs['loss_lambda'])

        # Move to device
        self.to(self.device)

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

    def encode_texts(
            self, texts, return_encoding_time=False,
            output_hidden_states=False
        ):
        max_length = self.configs['max_length']
        if (not self.training) or (not self.all_names_cache.initialized):
            # Do actual tokenization
            toks = self.tokenizer.batch_encode_plus(texts,
                                                    padding=True,
                                                    return_tensors='pt',
                                                    truncation=True,
                                                    max_length=max_length)
            if (not self.training) and SHOULD_SHUFFLE_DURING_INFERENCE:
                toks['input_ids'] = shuffle_input_ids(toks).to(self.device)
        else:
            # Use the cache
            toks = self.all_names_cache.encode(texts)

        toks = toks.to(self.device)
        start_encoding_time = time.time()
        outputs = \
            self.transformer(**toks, output_hidden_states=output_hidden_states)

        # Hidden states (if output_hidden_states is True)
        hidden_states = []
        if output_hidden_states:
            layer_outputs = list(outputs[2]) # including the embedding layer
            for layer in PEERS_LAYERS:
                hidden_states.append(layer_outputs[layer])

        # Main Representation
        reps = outputs[0][:, 0, :]
        if output_hidden_states:
            assert(torch.all(torch.eq(hidden_states[-1][:, 0, :], reps)))
        if self.ffnn: reps = self.ffnn(reps)
        encoding_time = time.time() - start_encoding_time

        if return_encoding_time:
            return reps, [toks, hidden_states], encoding_time
        return reps, [toks, hidden_states]

    def generate_candidates(self, instances, ontology):
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

class CrossAttentionEncoderModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model_type = RERANKER

        self.transformer = AutoModel.from_pretrained(configs['transformer'])
        self.transformer.config.gradient_checkpointing  = configs['gradient_checkpointing']
        self.tokenizer = AutoTokenizer.from_pretrained(configs['transformer'], use_fast=True)

        # Linear layer for scoring
        self.linear = nn.Linear(self.transformer.config.hidden_size, 1)

        # Loss Function
        self.loss_fct = nn.BCEWithLogitsLoss()

        # Move to device
        self.to(self.device)


    def forward(self, instances, ontology, is_training):
        configs = self.configs
        candidates = [inst.candidate_names[:configs['topk']] for inst in instances]
        assert(len(instances) == len(candidates))
        self.train() if is_training else self.eval()
        nb_instances = len(instances)
        nb_candidates_per_instance = len(candidates[0])

        # Encode mention-candidate pairs
        text_pairs, SEP = [], self.tokenizer.sep_token
        for ix, inst in enumerate(instances):
            # Each c should be OntologyNameEntry
            for c in candidates[ix]:
                term = inst.mention['term']
                l_ctx, r_ctx = inst.left_context, inst.right_context
                if configs['include_context']:
                    full_ctx = l_ctx + f' {SEP} ' + term + f' {SEP} ' + r_ctx
                    text_pairs.append((full_ctx, c.name_str))
                else:
                    text_pairs.append((term, c.name_str))
        reps = self.encode_texts(text_pairs)[0]

        # Predict raw scores
        scores = self.linear(reps).squeeze()
        scores = scores.view(nb_instances, nb_candidates_per_instance)

        # Extract predictions
        preds, candidate_names, candidate_dists = [], [], []
        sorted_indexes = torch.argsort(scores, dim=1, descending=True)
        for i in range(nb_instances):
            if instances[i].should_be_reranked:
                cur_preds, cur_candidate_names, cur_candidate_dists = [], [], []
                cur_candidates = candidates[i]
                for j in range(nb_candidates_per_instance):
                    pred_eid = cur_candidates[sorted_indexes[i, j]].entity_id
                    if not pred_eid in cur_preds: cur_preds.append(pred_eid)
                    cur_candidate_names.append(cur_candidates[sorted_indexes[i, j]])
                    cur_candidate_dists.append(tolist(1.0-scores[i, sorted_indexes[i, j]]))
                preds.append(cur_preds)
                candidate_names.append(cur_candidate_names)
                candidate_dists.append(cur_candidate_dists)
                # Sanity checks
                assert(cur_candidate_dists[0] <= cur_candidate_dists[1])
                assert(cur_candidate_dists[1] <= cur_candidate_dists[2])
            else:
                preds.append(instances[i].candidate_entities)
                candidate_names.append(instances[i].candidate_names)
                candidate_dists.append(instances[i].candidate_distances)

        # Loss (if trainig)
        if not is_training:
            loss = 0
        else:
            labels = self.extract_labels(instances, candidates).to(self.device)
            loss = self.loss_fct(scores.view(-1), labels.view(-1))

        return loss, preds, candidate_names, candidate_dists

    def extract_labels(self, instances, candidates):
        assert(len(instances) == len(candidates))
        nb_instances = len(instances)
        nb_candidates_per_instance = len(candidates[0])
        labels = [[0 for i in range(nb_candidates_per_instance)] for j in range(nb_instances)]
        for i in range(nb_instances):
            for j in range(nb_candidates_per_instance):
                if instances[i].mention['entity_id'] == candidates[i][j].entity_id:
                    labels[i][j] = 1
        labels = torch.FloatTensor(labels)
        return labels

    def encode_texts(self, texts):
        toks = self.tokenizer.batch_encode_plus(texts,
                                                padding=True,
                                                return_tensors='pt',
                                                truncation=True,
                                                max_length=self.configs['max_length'])
        toks = toks.to(self.device)
        outputs = self.transformer(**toks)
        reps = outputs[0][:, 0, :]
        return reps,

class FullModel(BaseModel):
    def __init__(self, cg_configs, rr_configs=None):
        #assert(cg_configs['no_cuda'] == rr_configs['no_cuda'])
        BaseModel.__init__(self, {'no_cuda': cg_configs['no_cuda']})
        self.cg_configs = cg_configs
        self.rr_configs = rr_configs

        # Initialize a DualBertEncodersModel (candidates generator)
        if not cg_configs['online_kd']:
            if cg_configs['lightweight']:
                print('class LightWeightModel')
                self.cg = LightWeightModel(cg_configs)
            else:
                print('class DualBertEncodersModel')
                self.cg = DualBertEncodersModel(cg_configs)
        else:
            print('class EncodersModelWithOnlineKD')
            self.cg = EncodersModelWithOnlineKD(cg_configs)

        # Initialize a CrossAttentionEncoderModel
        if rr_configs:
            self.rr = CrossAttentionEncoderModel(rr_configs)

        # Move to device
        self.to(self.device)

    def reload_cg(self, path):
        ckpt = torch.load(path, map_location=self.cg.device)
        self.cg.load_state_dict(ckpt['model_state_dict'], strict=False)
        print('Reloaded the candidate generator')

    def reload_rr(self, path):
        ckpt = torch.load(path, map_location=self.rr.device)
        self.rr.load_state_dict(ckpt['model_state_dict'], strict=False)
        print('Reloaded the reranker')

    def build_ontology_index(self, ontology):
        ontology.build_index(self.cg, 256)

# DummyModel (for debugging)
class DummyOntologyNameEntry:
    def __init__(self, name_str, entity_id):
        self.name_str = name_str.replace('+', '|')
        self.entity_id = entity_id.replace('+', '|')   # id of the entity that the name is associated with

class DummyModel(BaseModel):
    def __init__(self, configs):
        BaseModel.__init__(self, configs)
        self.model_type = DUMMY_MODEL
        self.configs = configs
        self.dataset = configs['dataset']
        assert(configs['dataset'] in [BC5CDR_C, NCBI_D])

        # Read predictions
        fp = join(join(BASE_RESOURCES_DIR, 'biosyn_preds'), f'{self.dataset}.json')
        with open(fp, 'r') as f:
            data = json.loads(f.read())

        # Process queries
        cache = {}
        for q in data['queries']:
            for m in q['mentions']:
                query_term, query_golden_cui = m['mention'], m['golden_cui']
                query_candidates = m['candidates']
                cache[(query_term, query_golden_cui)] = query_candidates
                assert(not '|' in query_term)
                assert(not '+' in query_term)

        self.cache = cache


    def forward(self, instances, ontology, is_training):
        assert(not is_training)
        all_entity_ids = set(ontology.all_entity_ids)
        candidate_entities, candidate_names, candidate_dists = [], [], []
        for inst in instances:
            cur_candidate_entites, cur_candidate_names, cur_candidate_dists = [], [], []
            if not '|' in inst.mention['term']:
                query_term = inst.mention['term']
                query_golden_cui = inst.mention['entity_id']
                cached_preds = self.cache[(query_term, query_golden_cui)]
                for p in cached_preds:
                    if not p['cui'] in cur_candidate_entites:
                        cur_candidate_entites.append(p['cui'])
                    cur_candidate_names.append(DummyOntologyNameEntry(p['name'], p['cui']))
                    assert(p['cui'].replace('+', '|') in all_entity_ids)
            candidate_entities.append(cur_candidate_entites)
            candidate_names.append(cur_candidate_names)
            candidate_dists.append(cur_candidate_dists)
        return 0, candidate_entities, candidate_names, candidate_dists
