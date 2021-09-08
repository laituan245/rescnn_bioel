import os
import json
import networkx as nx

from data import *

def extract_ontology(concept_fp, description_fp, output_fp):
    """
        concept_fp: Path to the concept file of SNOMED CT
        description_fp: Path to the description file of SNOMED CT
        output_fp: Path to the json file where the extractd ontology is written to
    """
    # Read the concept file
    entity_ids = set()
    with open(concept_fp, 'r', encoding='utf-8') as f:
        f.readline() # Skip the first line
        for line in f:
            es = line.strip().split('\t')
            concept_id, effective_time, active = es[:3]
            year, month = int(effective_time[:4]), int(effective_time[4:6])
            if int(active) == 1: entity_ids.add(concept_id)
    print('[SNOMED CT] Number of active entities: {}'.format(len(entity_ids)))

    # Read the description file
    entity2names = {}
    with open(description_fp, 'r', encoding='utf-8') as f:
        f.readline() # Skip the header line
        for line in f:
            es = line.strip().split('\t')
            description_id, _, active, _, concept_id, lang, typeid, name, _ = es
            nametype = 'primary' if typeid == '900000000000003001' else 'synonym/secondary'
            if int(active) == 1 and (concept_id in entity_ids) and lang == 'en':
                if not concept_id in entity2names: entity2names[concept_id] = set()
                entity2names[concept_id].add((nametype, name))
    assert(len(entity2names) == len(entity_ids))

    # Convert names set to names list
    for entity in entity2names:
        entity2names[entity] = list(entity2names[entity])

    # Sanity checks
    print('Running saniy checks')
    # Check if each entiy has exactly 1 primary name
    total_names, checked = 0, 0
    for entity in entity2names:
        names = entity2names[entity]
        primary_ctx = 0
        for n in names:
            if n[0] == 'primary': primary_ctx += 1
            total_names += 1
        assert(primary_ctx >= 1)
        checked += 1
        if checked % 100000 == 0: print('Checked: {}'.format(checked))
    print('[SNOMED CT] Number of names: {}'.format(total_names))

    # Output file
    print('Writing to output file')
    with open(output_fp, 'w+', encoding='utf-8') as f:
        json.dump(entity2names, f)

def clean_ontology():
    # Load data
    train, dev, test, ontology = load_data(COMETA)
    all_eids_in_dataset = set()
    all_insts = train.items + dev.items + test.items
    for inst in all_insts:
        all_eids_in_dataset.add(inst.mention['entity_id'])
    # Build namestr2eids = {}
    namestr2eids = {}
    for n in ontology.name_list:
        name_str, entity_id = n.name_str, n.entity_id
        if not name_str in namestr2eids: namestr2eids[name_str] = set()
        namestr2eids[name_str].add(entity_id)
    # Build eids_to_be_removed
    eids_to_be_removed = set()
    for name_str in namestr2eids:
        eids = namestr2eids[name_str]
        if len(eids) == 1: continue
        for eid in eids:
            if not eid in all_eids_in_dataset:
                eids_to_be_removed.add(eid)
    print(f'To remove {len(eids_to_be_removed)} entities')
    # output new ontology
    with open(SNOMEDCT_FP, 'r') as f:
        data = json.loads(f.read())
    for eid in eids_to_be_removed:
        del data[eid]
    with open('resources/ontologies/cleaned_snomedct.json', 'w+') as f:
        f.write(json.dumps(data))


# Code from https://github.com/cambridgeltl/cometa
# Column indices of the interesting fields in SNOMED's
# Description and Relationship files
_SNOMED_TERM_FIELD_ID = 0  # term activity (deprecated terms will have active = 0)
_SNOMED_TERM_FIELD_ACTIVE = 2  # term activity (deprecated terms will have active = 0)
_SNOMED_DESC_FIELD_ACTIVE = 2  # definition activity (deprecated definition will have active = 0)
_SNOMED_DESC_FIELD_ID = 4  # term id
_SNOMED_DESC_FIELD_DEF = 7  # term description
_SNOMED_REL_FIELD_ACTIVE = 2  # relationship activity (deprecated relationship will have active = 0)
_SNOMED_REL_FIELD_ID = 7  # relationship id
_SNOMED_REL_FIELD_SOURCE = 4  # relationship source node
_SNOMED_REL_FIELD_TARGET = 5  # relationship target node

# Useful relationship IDs
_SNOMED_REL_IS_A = '116680003'


class Snomed:
    def __init__(self, snomed_path='SnomedCT_International', release_id='20190731', taxonomy=False):
        self.snomed_path = snomed_path
        self.release_id = release_id
        self.definition_index = {}
        self.index_definition = {}
        self.graph = None
        self.taxonomy = taxonomy

    def load_snomed(self):

        # init graph
        if self.taxonomy:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        # set of active nodes
        nodes = set()

        # load active nodes
        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Concept_Snapshot_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)
            for line in f:
                concept = line.split('\t')
                if int(concept[_SNOMED_TERM_FIELD_ACTIVE]):
                    nodes.add(concept[_SNOMED_TERM_FIELD_ID])

        # load definitions
        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Description_Snapshot-en_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)

            for line in f:
                definition = line.split('\t')
                cid = definition[_SNOMED_DESC_FIELD_ID]  # concept id
                cdesc = definition[
                    _SNOMED_DESC_FIELD_DEF]  # concept description

                # load only active defintions and only the first definition
                if cid in nodes and (int(definition[_SNOMED_DESC_FIELD_ACTIVE])
                                     ):
                    # add the first definition as an attribute
                    if (cid not in self.graph):
                        self.graph.add_node(cid, desc=cdesc)
                    # and put the first and the others in the indices
                    self.definition_index[cdesc] = cid
                    if cid not in self.index_definition:
                        self.index_definition[cid] = [cdesc]
                    else:
                        self.index_definition[cid].append(cdesc)
                # fi
            # for
        # with

        with open(os.path.join(
                self.snomed_path,
                f'Snapshot/Terminology/sct2_Relationship_Snapshot_INT_{self.release_id}.txt'
        ),
                  mode='r',
                  encoding='utf8') as f:

            # skip header
            next(f)

            for line in f:
                rel = line.split('\t')
                if int(rel[_SNOMED_REL_FIELD_ACTIVE]):
                    # load only IS-A relationships in taxonomy mode
                    if (self.taxonomy and (rel[_SNOMED_REL_FIELD_ID] != _SNOMED_REL_IS_A)):
                        continue

                    self.graph.add_edge(
                        rel[_SNOMED_REL_FIELD_TARGET],rel[_SNOMED_REL_FIELD_SOURCE])
                # fi
            # for
        # with


    def __contains__(self, index):
        """
        Wrapper for `networkx.Graph.has_node()`
        """

        if type(index) != str:
            index = str(index)


        return index in self.graph


    def __getitem__(self, index):
        """
        Utility method to access nodes of SNOMED more easily.
        Allows using strings or ids as indices.

        Example:
        ```
        > snomed = Snomed('path/to/snomed')
        > snomed.load_snomed()
        > snomed['774007']
        {'desc': 'Head and neck'}
        > snomed[774007]
        {'desc': 'Head and neck'}
        ```
        """

        if type(index) != str:
            index = str(index)

        return self.graph.nodes[index]


    def predecessors(self, index):
        """
        Wrapper of networkx.digraph.predecessors()
        """

        if not self.taxonomy:
            raise RuntimeError('Ancestry is supported only in taxonomy mode')

        if type(index) != str:
            index = str(index)

        return list(self.graph.predecessors(index))

    def successors(self, index):

        if not self.taxonomy:
            raise RuntimeError('Ancestry is supported only in taxonomy mode')

        """
        Wrapper of networkx.digraph.successors()
        """
        if type(index) != str:
            index = str(index)

        return list(self.graph.successors(index))

    def distance(self, source, target):
        """
        Computes the distance between two nodes.
        """

        if type(source) != str:
            source = str(source)

        if type(target) != str:
            target = str(target)

        if nx.has_path(self.graph,source=source, target=target):
            return nx.shortest_path_length(self.graph,source=source,target=target)
        else:
            return nx.shortest_path_length(self.graph,source=target,target=source)


    def is_ancestor(self, source, target):
        """
        Returns True if `source` is an ancestor of `target` in the SNOMED taxonomy.
        """

        if not self.taxonomy:
            raise RuntimeError('Ancestry is supported only in taxonomy mode')

        if type(source) != str:
            source = str(source)

        if type(target) != str:
            target = str(target)

        return nx.has_path(self.graph,source=source, target=target)


    def safe_distance(self, source, target):
        """
        Computes the distance between two nodes. If there's not path between the source
        and target node, returns -1.
        """

        if type(source) != str:
            source = str(source)

        if type(target) != str:
            target = str(target)

        if nx.has_path(self.graph,source=source, target=target):
            return nx.shortest_path_length(self.graph,source=source,target=target)
        elif nx.has_path(self.graph,source=target, target=source):
            return nx.shortest_path_length(self.graph,source=target,target=source)
        else:
            return -1

    # build snomed surface->node_id dict
    def build_surface_to_snomed_id(self):
        sf2id = {}
        for node_id in self.graph.nodes:
            sfs = self.index_definition[node_id]
            for sf in sfs:
                sf2id[sf.lower()] = int(node_id)
        return sf2id
