from constants import *
from transformers import *
from data import load_data
from argparse import ArgumentParser
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--dataset', default=COMETA, choices=DATASETS)
    args = parser.parse_args()
    dataset_name = args.dataset

    # Load data
    train, dev, test, ontology = load_data(dataset_name)
    print(f'Loaded {dataset_name}')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    print(f'Loaded tokenizer')

    # Mentions Length Analysis
    mention_lengths = []
    for inst in train.items:
        term = inst.mention['term']
        tokens = tokenizer.tokenize(term)
        mention_lengths.append(len(tokens))

    # Plot histograms
    plt.hist(mention_lengths)
    plt.title('Mention Length')
    plt.savefig(f'figures/lengths/mentions_{dataset_name}.png', bbox_inches='tight')

    # Mentions Length > 5
    ctx_greater_than_5 = len([l for l in mention_lengths if l > 5])
    percent = 100 * ctx_greater_than_5 / len(mention_lengths)
    print(f'Percent of mentions with length > 5: {percent}%')

    # Ontology Length Analysis
    ontology_lengths = []
    for n in ontology.name_list:
        ontology_lengths.append(len(tokenizer.tokenize(n.name_str)))

    # Plot histograms
    plt.hist(ontology_lengths)
    plt.title('Entity Names Length')
    plt.savefig(f'figures/lengths/entity_names_{dataset_name}.png', bbox_inches='tight')

    # Entity names with length > 5
    ctx_greater_than_5 = len([l for l in ontology_lengths if l > 5])
    percent = round(100 * ctx_greater_than_5 / len(ontology_lengths), 3)
    print(f'Percent of entity names with length > 5: {percent}%')
