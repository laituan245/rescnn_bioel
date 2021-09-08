from utils import *
from constants import *
from os.path import join
from data.base import DataInstance

def load_medmentions_dataset(base_path, ontology):
    example_id, total_mentions, unmatchable_mentions = 0, 0, {}
    train_mentions, train_skipped = set(), 0
    train, dev, test = [], [], []
    all_entity_ids = set(ontology.all_entity_ids)
    for eid in all_entity_ids: assert(not '|' in eid)
    for split in [TRAIN, DEV, TEST]:
        data_fp = join(base_path, '{}.txt'.format(split))
        with open(data_fp, 'r', encoding="utf8", errors='ignore') as f:
            for line in f:
                es = line.strip().split('\t')
                assert(len(es) == 3)
                doc_id, term, entity_id = es
                mention = {'term': term, 'entity_id': entity_id}
                assert(mention['entity_id'].count('|') == 0 and mention['entity_id'].count('+') == 0)
                assert(mention['term'].count('|') == 0)
                total_mentions += 1
                if not entity_id in all_entity_ids:
                    unmatchable_mentions[split] = unmatchable_mentions.get(split, 0) + 1
                    if split == TRAIN:
                        train_skipped += 1
                        continue
                inst = DataInstance(example_id, '', mention)
                if split == TRAIN:
                    if (mention['term'], mention['entity_id']) in train_mentions:
                        train_skipped += 1
                        continue
                    train.append(inst)
                    train_mentions.add((mention['term'], mention['entity_id']))
                if split == DEV: dev.append(inst)
                if split == TEST: test.append(inst)
                example_id += 1
    print(f'Unmatchable mentions (w.r.t ontology): {unmatchable_mentions}')
    print(f'Total number of mentions: {total_mentions}')
    print(f'Number of train examples skipped: {train_skipped}')
    return train, dev, test
