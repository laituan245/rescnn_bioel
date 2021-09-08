import torch
import random
import nltk
import numpy as np
import logging
logger = logging.getLogger(__name__)

from constants import *
from transformers import *
from argparse import ArgumentParser
from data.base import Ontology, DataInstance, PretrainingPositivePairs
from data.bart import BartDataset
from datasets import load_metric
from utils import get_n_params, create_dir_if_not_exist


def train_bart(args):
    # Prepare config, tokenizer, and model
    config = AutoConfig.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir, use_fast=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, config=config, cache_dir=args.cache_dir
    )
    model.config.gradient_checkpointing  = args.gradient_checkpointing
    if args.gradient_checkpointing:
        model.config.use_cache = False
    print(f'Prepared config, tokenizer, and model ({get_n_params(model)} params)')

    # Prepare train_dataset and val_dataset
    pairs = PretrainingPositivePairs(args.file_path).positive_pairs
    random.shuffle(pairs)
    val_size = min(10000, int(0.1 * len(pairs)))
    train_pairs, val_pairs = pairs[:-val_size], pairs[-val_size:]
    train_dataset = BartDataset(train_pairs, args.max_length, tokenizer)
    eval_dataset = BartDataset(val_pairs, args.max_length, tokenizer)
    print(f'Train Size: {len(train_dataset)} | Val Size: {len(eval_dataset)}')
    eval_steps = int(len(train_pairs) / (5 * args.batch_size))

    # Metric
    metric = load_metric('rouge')

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ['\n'.join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ['\n'.join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result['gen_len'] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    # Create TrainingArguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        learning_rate=float(args.learning_rate),
        weight_decay=0.01,
        num_train_epochs=int(args.num_train_epochs),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.batch_size),
        predict_with_generate=True,
        evaluation_strategy='steps', eval_steps=eval_steps,
        load_best_model_at_end=True
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics['train_samples'] = len(train_dataset)

    trainer.log_metrics('train', metrics)
    trainer.save_metrics('train', metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(max_length=args.max_length, metric_key_prefix='eval')
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

    print(metrics)

    return metrics

if __name__ == "__main__":
    # Parse argument
    parser = ArgumentParser()
    parser.add_argument('--model_name_or_path', default='facebook/bart-base')
    parser.add_argument('--file_path', default='/shared/nas/data/m1/tuanml/biolinking/data/umls/pretrain_pairs_without_trivials.txt')
    parser.add_argument('--cache_dir', default='/shared/nas/data/m1/tuanml2/cache/')
    parser.add_argument('--output_dir', default='/shared/nas/data/m1/tuanml2/bart_trained')
    parser.add_argument('--learning_rate', default=5e-5)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--max_length', default=25)
    parser.add_argument('--num_train_epochs', default=2)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    args = parser.parse_args()
    args.max_length = int(args.max_length)
    args.learning_rate = float(args.learning_rate)
    args.num_train_epochs = int(args.num_train_epochs)
    args.batch_size = int(args.batch_size)
    create_dir_if_not_exist(args.output_dir)

    train_bart(args)
