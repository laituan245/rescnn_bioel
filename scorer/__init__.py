from utils import *
from tqdm import tqdm
from constants import *
from data.base import DataInstance

def check_labels(gt_entity_ids, topk_entity_ids):
    crtx_top1, crtx_top5, crtx_top10, crtx_top20 = 0, 0, 0, 0
    for gt_entity_id in gt_entity_ids:
        for i in range(min(20, len(topk_entity_ids))):
            assert(not '+' in topk_entity_ids[i])
            pred_entity_ids = topk_entity_ids[i].split('|')
            if gt_entity_id in pred_entity_ids:
                if i < 1: crtx_top1 = 1
                if i < 5: crtx_top5 = 1
                if i < 10: crtx_top10 = 1
                if i < 20: crtx_top20 = 1
    return crtx_top1, crtx_top5, crtx_top10, crtx_top20

def evaluate(model, dataset, ontology, configs, return_correct_cases=False):
    # Apply the model
    model_infeference_time = \
        apply_model(model, dataset, ontology, configs['batch_size'])

    correct_cases = set()
    total_ctx, correct_top1, correct_top5, correct_top10, correct_top20 = 0, 0, 0, 0, 0
    for inst in dataset.items:
        assert(not '+' in inst.mention['entity_id'])
        gt_entity_ids = inst.mention['entity_id'].split('|')
        topk_entity_ids = [n.entity_id for n in inst.candidate_names]

        # Update cur counts
        if inst.mention['term'].count('|') == 0:
            # Singleton
            cur_crtx_1, cur_crtx_5, cur_crtx_10, cur_crtx_20 = \
                                check_labels(gt_entity_ids, topk_entity_ids)
        else:
            # Composite
            composite_term = inst.mention['term']
            mini_terms = composite_term.split('|')
            mini_insts = []
            for mini_term in mini_terms:
                mini_mention = {'term': mini_term, 'entity_id': '|'.join(gt_entity_ids)}
                # Build a mini_inst
                mini_inst = DataInstance(inst.id, inst.context, mini_mention)
                mini_inst.candidate_entities = inst.candidate_entities
                mini_inst.candidate_names = inst.candidate_names
                mini_inst.candidate_distances = inst.candidate_distances
                try:
                    mini_inst.should_be_reranked = inst.should_be_reranked
                except:
                    pass
                # Update mini_insts list
                mini_insts.append(mini_inst)
            apply_model(model, AugmentedList(mini_insts), ontology, len(mini_insts), disable_tqdm=True)
            # Check if all mini term is predicted correctly
            cur_crtx_1, cur_crtx_5, cur_crtx_10, cur_crtx_20 = 1, 1, 1, 1
            for mini_inst in mini_insts:
                mini_inst_topk_entity_ids = [n.entity_id for n in mini_inst.candidate_names]
                mini_crtx_1, mini_crtx_5, mini_crtx_10, mini_crtx_20 = \
                                    check_labels(gt_entity_ids, mini_inst_topk_entity_ids)
                cur_crtx_1 &= mini_crtx_1
                cur_crtx_5 &= mini_crtx_5
                cur_crtx_10 &= mini_crtx_10
                cur_crtx_20 &= mini_crtx_20

        # Update global counts
        if cur_crtx_1 > 0:
            correct_cases.add(inst.id)
        correct_top1 += cur_crtx_1
        correct_top5 += cur_crtx_5
        correct_top10 += cur_crtx_10
        correct_top20 += cur_crtx_20
        total_ctx += 1

    if configs['dataset'] == COMETA and COMETA_SETTING == STRATIFIED_SPECIFIC:
        assert(total_ctx == 13441 or total_ctx == 2205 or total_ctx == 4369)

    if total_ctx == 0: return {}

    eval_results = {
        'top1_accuracy': round(correct_top1 / total_ctx, 5),
        'top5_accuracy': round(correct_top5 / total_ctx, 5),
        'top10_accuracy': round(correct_top10 / total_ctx, 5),
        'top20_accuracy': round(correct_top20 / total_ctx, 5)
    }

    if return_correct_cases:
        return eval_results, correct_cases
    print(f'model_infeference_time = {model_infeference_time}')
    return eval_results
