import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from os.path import exists, join
from models.base import *
from models.model import *
from models.helpers import *

# Get total number of parameters in a model
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class EncodersModelWithOnlineKD(DualBertEncodersModel):
    def __init__(self, configs):
        assert(configs['online_kd'])
        configs['feature_proj'] = False
        DualBertEncodersModel.__init__(self, configs)
        self.online_kd = configs['online_kd']
        self.child_branch_exit = False
        self.selected_child_branch = None

        # Online KD with Multiple Peers (if enabled)
        if self.online_kd:
            total_nb_params_increased = 0

            # FFNNs for feature projections (typically 768 -> 256)
            if configs['enable_branch_ffnns']:
                # FFNNs for Hidden States
                self.feature_size = configs['feature_size']
                branch_ffnns = []
                for layer in PEERS_LAYERS:
                    hidden_size = (self.transformer_feature_size + self.feature_size) // 2
                    branch_ffnns.append(
                        FFNNModule(
                            self.transformer_feature_size,
                            [hidden_size],
                            self.feature_size
                    ))
                self.branch_ffnns = nn.ModuleList(branch_ffnns)
                branch_ffns_params = get_n_params(self.branch_ffnns[0])
                print(f'Nb params in one branch ffnn is: {branch_ffns_params}')
                total_nb_params_increased += len(self.branch_ffnns) * branch_ffns_params

            # Transformer modules for each student branch
            student_transformers = []
            for layer in PEERS_LAYERS:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.feature_size,
                    nhead=8, dim_feedforward=512
                )
                _encoder_norm = nn.LayerNorm(self.feature_size, eps=1e-5)
                _transformer = nn.TransformerEncoder(encoder_layer, 2, _encoder_norm)
                student_transformers.append(_transformer)
            self.student_transformers = nn.ModuleList(student_transformers)
            student_transformers_params = get_n_params(self.student_transformers[0])
            print(f'Nb params in one student transformer: {student_transformers_params}')
            total_nb_params_increased += len(self.student_transformers) * student_transformers_params

            # PositionalEncoding and Transformer Encoder for ensemble
            self.ensemble_pe = PositionalEncoding(self.feature_size, max_len=len(PEERS_LAYERS))
            ensemble_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.feature_size,
                nhead=8, dim_feedforward=512
            )
            ensemble_encoder_norm = nn.LayerNorm(self.feature_size, eps=1e-5)
            self.ensemble_transformer = nn.TransformerEncoder(ensemble_encoder_layer, 2, ensemble_encoder_norm)
            total_nb_params_increased += get_n_params(self.ensemble_pe)
            total_nb_params_increased += get_n_params(self.ensemble_transformer)

            print(f'total_nb_params_increased: {total_nb_params_increased}')

            # KD loss function
            self.kd_loss_fct = nn.MSELoss()

        # Move to device
        self.to(self.device)

    def forward(self, instances, ontology, is_training):
        if (not self.online_kd) or (not is_training):
            return DualBertEncodersModel.forward(self, instances, ontology, is_training)

        self.train() if is_training else self.eval()
        loss = 0.0

        # Queries' representations and hidden states
        query_texts = [inst.mention['term'] for inst in instances]
        query_reps, query_hidden_states = self.encode_texts(query_texts)

        # Candidates' representations and hidden states
        candidates_eids, candidates_texts = self.generate_candidates(instances, ontology)
        candidates_reps, candidate_hidden_states = self.encode_texts(candidates_texts)

        # candidate_labels
        all_mentions_eids = list(set(candidates_eids))
        candidate_labels = [all_mentions_eids.index(eid) for eid in candidates_eids]
        candidate_labels = torch.LongTensor(candidate_labels).to(self.device)

        # query_labels
        query_labels = [all_mentions_eids.index(inst.mention['entity_id']) for inst in instances]
        query_labels = torch.LongTensor(query_labels).to(self.device)

        # all_labels and all_reps
        all_labels = torch.cat([query_labels, candidate_labels], dim=0)
        all_reps = torch.cat([query_reps, candidates_reps], dim=0)

        # Compute loss between ensemble's reprs and gt labels
        ensemble_miner_output = self.miner_fct(all_reps, all_labels)
        loss += self.loss_fct(all_reps, all_labels, ensemble_miner_output)

        # Compute loss between each peer's reprs and gt labels
        peer_all_reps = []
        for ix, layer in enumerate(PEERS_LAYERS):
            peer_query_reps = query_hidden_states[:, ix, :]
            peer_candidate_reps = candidate_hidden_states[:, ix, :]
            peer_reps = torch.cat([peer_query_reps, peer_candidate_reps], dim=0)
            # Update loss
            peer_miner_output = self.miner_fct(peer_reps, all_labels)
            loss += self.loss_fct(peer_reps, all_labels, peer_miner_output)
            # Update peer_all_reps
            peer_all_reps.append(peer_reps)

        # Compute kd loss
        if self.configs['kd_loss_term_type'] == 'none':
            pass
        elif self.configs['kd_loss_term_type'] == 'individual':
            for ix, layer in enumerate(PEERS_LAYERS):
                loss += self.kd_loss_fct(peer_all_reps[ix], all_reps)
        elif self.configs['kd_loss_term_type'] == 'relational':
            ensemble_corr_matrix = self.compute_corr_matrix(all_reps)
            for ix, layer in enumerate(PEERS_LAYERS):
                peer_corr_matrix = self.compute_corr_matrix(peer_all_reps[ix])
                loss += self.kd_loss_fct(peer_corr_matrix, ensemble_corr_matrix)
        else:
            raise NotImplementedError

        return loss, [], [], []

    def compute_corr_matrix(self, reprs):
        corr_matrix = torch.matmul(reprs, reprs.T)
        return corr_matrix

    def encode_texts(self, texts):
        if not self.online_kd:
            return DualBertEncodersModel.encode_texts(self, texts)

        # Online KD with Multiple Peers
        _, [toks, hidden_states] = DualBertEncodersModel.encode_texts(self, texts, True)
        attention_mask = (1-toks['attention_mask']).to(self.device).bool()

        # Detach hidden states (except the last layer)
        # for i in range(len(hidden_states)-1):
        #     hidden_states[i] = hidden_states[i].detach()

        # FFNNs for Hidden States (~ Branches)
        if self.configs['enable_branch_ffnns']:
            for i in range(len(hidden_states)):
                hidden_states[i] = self.branch_ffnns[i](hidden_states[i])

        # Apply Student Transformers
        for i, _transformer in enumerate(self.student_transformers):
            hidden_states[i] = _transformer(
                torch.transpose(hidden_states[i], 0, 1),
                src_key_padding_mask=attention_mask
            )
            hidden_states[i] = torch.transpose(hidden_states[i], 0, 1)
            assert(hidden_states[i].size()[1] <= self.configs['max_length'])
            assert(hidden_states[i].size()[-1] == self.feature_size)


        # Extract only vectors at the first position ([CLS] token)
        for i in range(len(hidden_states)):
            hidden_states[i] = hidden_states[i][:, 0, :]

        # Normalization
        for i in range(len(hidden_states)):
            hidden_states[i] = F.normalize(hidden_states[i], dim=-1, p=2)

        # Child branch early exit (if enabled)
        if self.child_branch_exit and (not self.selected_child_branch is None):
            branch_idx = PEERS_LAYERS.index(self.selected_child_branch)
            reps = hidden_states[branch_idx]
            all_hidden_states = torch.cat([h.unsqueeze(1) for h in hidden_states], dim=1)
            return reps, all_hidden_states

        # Concatenate hidden states into all_hidden_states
        hidden_states = [h.unsqueeze(1) for h in hidden_states]
        all_hidden_states = torch.cat(hidden_states, dim=1)

        # Compute ensemble reps
        ensemble_rep = self.ensemble_pe(torch.transpose(all_hidden_states, 0, 1))
        ensemble_rep = self.ensemble_transformer(ensemble_rep)
        ensemble_rep = torch.transpose(ensemble_rep, 0, 1)[:, -1, :]
        ensemble_rep = F.normalize(ensemble_rep, dim=-1, p=2)

        return ensemble_rep, all_hidden_states

    def enable_child_branch_exit(self, branch):
        assert(branch in PEERS_LAYERS)
        self.child_branch_exit = True
        self.selected_child_branch = branch

    def disable_child_branch_exit(self):
        self.child_branch_exit = False
        self.selected_child_branch = None
