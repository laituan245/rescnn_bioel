from utils import *
from constants import *
from os.path import join
from data.base import DataInstance

def load_cometa_dataset(base_path, ontology):
    all_entity_ids = set(ontology.all_entity_ids)
    all_ontology_terms = set([n.name_str for n in ontology.name_list])
    for eid in all_entity_ids: assert(not '|' in eid)
    base_path = join(base_path, COMETA_SETTING)
    train, dev, test = [], [], []
    for split in [TRAIN, DEV, TEST]:
        data_fp = join(base_path, '{}.csv'.format(split))
        with open(data_fp, 'r', encoding='utf-8') as f:
            f.readline() # Skip the first line
            for line in f:
                es = line.strip().split('\t')
                example_id, term, _, general_id, _, specific_id, context = es[:7]
                assert(general_id in all_entity_ids)
                assert(specific_id in all_entity_ids)
                mention = {'term': term}
                if COMETA_SETTING.endswith('general'): mention['entity_id'] = general_id
                if COMETA_SETTING.endswith('specific'): mention['entity_id'] = specific_id
                assert(mention['entity_id'].count('|') == 0 and mention['entity_id'].count('+') == 0)
                assert(mention['term'].count('|') == 0 and mention['term'].count('+') == 0)
                inst = DataInstance(example_id, context, mention)
                if COMETA_REMOVE_EASY_CASES:
                    if inst.mention['term'] in all_ontology_terms:
                        # Skip easy cases
                        continue
                if split == TRAIN: train.append(inst)
                if split == DEV: dev.append(inst)
                if split == TEST: test.append(inst)

    # Sanity checks
    # Data stats are from https://arxiv.org/pdf/2010.03295.pdf
    if COMETA_SETTING == STRATIFIED_SPECIFIC and (not COMETA_REMOVE_EASY_CASES):
        assert(len(train) == 13441)
        assert(len(dev) == 2205)
        assert(len(test) == 4369)
    # Stats
    train_entities = set([inst.mention['entity_id'] for inst in train])
    dev_entities = set([inst.mention['entity_id'] for inst in dev])
    train_dev_entities = train_entities.union(dev_entities)
    test_entities = set([inst.mention['entity_id'] for inst in test])
    overlap = test_entities.intersection(train_entities.union(dev_entities))
    print('Statistics of unique entities')
    print(f'train+dev size: {len(train_dev_entities)} | test size: {len(test_entities)}')
    print(f'overlap size (between train+dev and test): {len(overlap)}')

    return train, dev, test
