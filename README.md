# Biomedical Entity Linking

This repo provides the code for the paper [BERT might be Overkill: A Tiny but Effective Biomedical Entity Linker based on Residual Convolutional Neural Networks](https://arxiv.org/pdf/2109.02237.pdf) (EMNLP 2021 Findings).

Basic running instructions
```
pip install -r requirements.txt
python cg_trainer.py --dataset bc5cdr-chemical
```

Please refer to the file `constants.py` for the list of all supported datasets.
Note that for COMETA, you need to download the dataset from https://github.com/cambridgeltl/cometa.

Note that for ncbi-disease, bc5cdr-disease, and bc5cdr-chemical, we follow the protocol of [BioSyn](https://github.com/dmis-lab/BioSyn). We use development (dev) set to search the hyperparameters, and train on traindev (train+dev) set to report the final performance.

We are cleaning the codebase and we will add more running instructions soon.
