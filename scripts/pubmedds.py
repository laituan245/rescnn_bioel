import json
import scispacy
import random
import spacy

from utils import *
from os import listdir
from os.path import isfile, join

def extract_terms(base_dir='/shared/nas/data/m1/tuanml/bioie/data/pubmed_ds',
                  output_fp='pubmedds_terms.txt'):
    # scispacy
    nlp = spacy.load('en_core_sci_lg')
    print('Prepare scispacy pipeline')

    # Get the list of filenames
    filenames = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]
    filenames = [f for f in filenames if f.endswith('.txt')]

    # Process each file
    biomedical_terms = set()
    with tqdm(total=len(filenames), desc=f'Processing') as pbar:
        for filename in filenames:
            file_path = join(base_dir, filename)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    doc = nlp(data['text'].lower())
                    for x in doc.ents:
                        biomedical_terms.add(x.text)
                    pbar.set_postfix_str(f'Number Terms: {len(biomedical_terms)}')
            pbar.update(1)

    # Output
    with open(output_fp, 'w+') as f:
        for term in biomedical_terms:
            f.write('{}\n'.format(term))

