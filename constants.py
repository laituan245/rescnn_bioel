from os.path import dirname, join, realpath, isfile

TRAIN, DEV, TEST = 'train', 'dev', 'test'

# Basic Constants
BASE_PATH = dirname(realpath(__file__))
BASE_RESOURCES_DIR = join(BASE_PATH, 'resources')
BASIC_CONF_PATH = join(BASE_PATH, 'configs/exp.conf')
BASE_SAVE_PATH = 'trained'
BASE_SYNTHETIC_DATA_PATH = '/shared/nas/data/m1/tuanml/biolinking/synthetic_data/'
BASE_CLUSTERS_INFO_PATH = '/shared/nas/data/m1/tuanml/biolinking/data/clusters_info/'
FREQUENT_WORDLIST = 'resources/others/google-10000-english-no-swears.txt'
SHOULD_SHUFFLE_DURING_INFERENCE = False
PRETRAINED_LIGHTWEIGHT_VDCNN_MODEL = \
    '/shared/nas/data/m1/tuanml/biolinking/pretrained_models/pretraining_lightweight_vdcnn/model.pt'
PRETRAINED_LIGHTWEIGHT_CNN_TEXT_MODEL = \
    '/shared/nas/data/m1/tuanml/biolinking/pretrained_models/pretraining_lightweight_cnn_text/model.pt'

# Datasets
COMETA = 'cometa'
MEDMENTIONS = 'medmentions'
BC5CDR_C = 'bc5cdr-chemical'
BC5CDR_D = 'bc5cdr-disease'
NCBI_D = 'ncbi-disease'
DATASETS = [BC5CDR_C, BC5CDR_D, NCBI_D, COMETA, MEDMENTIONS]
USE_TRAINDEV = True

# Cometa Dataset
STRATIFIED_SPECIFIC = 'stratified_specific'
STRATIFIED_GENERAL = 'stratified_general'
ZEROSHOT_GENERAL = 'zeroshot_general'
COMETA_REMOVE_EASY_CASES = False
COMETA_SETTING = STRATIFIED_GENERAL

# Nametypes
NAME_PRIMARY = 'primary'
NAME_SECONDARY = 'synonym/secondary'
NAMETYPES = [NAME_PRIMARY, NAME_SECONDARY]

# Ontologies File Path
BASE_ONTOLOGY_DIR = 'resources/ontologies'
SNOMEDCT_FP = 'resources/ontologies/snomedct.json'
UMLS_2017AA_ACTIVE_FP = '/shared/nas/data/m1/tuanml/biolinking/data/umls/umls.2017AA.active.json'
if not isfile(UMLS_2017AA_ACTIVE_FP):
    # Local path
    UMLS_2017AA_ACTIVE_FP = 'umls.2017AA.active.json'
UMLS_2020AA_FULL_FP = '/shared/nas/data/m1/tuanml/biolinking/data/umls/umls.2020AA.full.json'
if not isfile(UMLS_2020AA_FULL_FP):
    # Local path
    UMLS_2020AA_FULL_FP = 'umls.2020AA.full.json'
    if not isfile(UMLS_2020AA_FULL_FP):
        # NCSA server
        UMLS_2020AA_FULL_FP = '/projects/bbqy/laituan245/el/data/umls.2020AA.full.json'

# Model Types
DUMMY_MODEL = 'dummy'
CANDIDATES_GENERATOR = 'cg'
PRETRAINING_MODEL = 'pm'
RERANKER = 'rr'

# Online KD with Multiple Peers
PEERS_LAYERS = [3, 6, 9, 12]

# Pretraining Dataset
UMLS_HARD_NEGATIVES_FP = '/shared/nas/data/m1/tuanml/biolinking/data/umls/hard_negatives.txt'
if not isfile(UMLS_HARD_NEGATIVES_FP):
    # Local path
    UMLS_HARD_NEGATIVES_FP = 'hard_negatives.txt'

# UMLS Pretrain Positive Pairs File Path
UMLS_PRETRAIN_POSITIVE_PAIRS = '/shared/nas/data/m1/tuanml/biolinking/data/umls/pretrain_positive_pairs.txt'
if not isfile(UMLS_PRETRAIN_POSITIVE_PAIRS):
    # Local path
    UMLS_PRETRAIN_POSITIVE_PAIRS = 'pretrain_positive_pairs.txt'
    if not isfile(UMLS_PRETRAIN_POSITIVE_PAIRS):
        # NCSA server
        UMLS_PRETRAIN_POSITIVE_PAIRS = '/projects/bbqy/laituan245/el/data/pretrain_positive_pairs.txt'
