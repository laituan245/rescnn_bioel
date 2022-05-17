# Biomedical Entity Linking

**For reproducing our results on COMETA, please use the branch [cometa](https://github.com/laituan245/rescnn_bioel/tree/cometa).** Basically, we turn on this mode called *hard_negatives_training* when experimenting with COMETA.


This repo provides the code for the paper [BERT might be Overkill: A Tiny but Effective Biomedical Entity Linker based on Residual Convolutional Neural Networks](https://arxiv.org/pdf/2109.02237.pdf) (EMNLP 2021 Findings).

Download the pretrained embedding layer from this [link](https://drive.google.com/file/d/1zQ8mt7JI0hJWK-ilhxE93RJ6EnWZsSmj/view?usp=sharing). And set this [line](https://github.com/laituan245/rescnn_bioel/blob/main/configs/exp.conf#L123) to the path of the downloaded file.

Basic running instructions
```
pip install -r requirements.txt
python cg_trainer.py --dataset bc5cdr-chemical
```

Please refer to the file `constants.py` for the list of all supported datasets.
Note that for COMETA, you need to download the dataset from https://github.com/cambridgeltl/cometa.

Note that for ncbi-disease, bc5cdr-disease, and bc5cdr-chemical, we follow the protocol of [BioSyn](https://github.com/dmis-lab/BioSyn). We use development (dev) set to search the hyperparameters, and train on traindev (train+dev) set to report the final performance.

We are cleaning the codebase and we will add more running instructions soon.
