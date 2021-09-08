import torch
from data.base import PretrainingPositivePairs

class BartDataset:
    def __init__(self, pairs, max_length, tokenizer):
        self.inputs = [p[0].name_str for p in pairs]
        self.targets = [p[1].name_str for p in pairs]
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        tokenizer, max_length = self.tokenizer, self.max_length
        _input, _target = self.inputs[idx], self.targets[idx]

        model_inputs = tokenizer([_input], max_length=max_length, padding='max_length', truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer([_target], max_length=max_length, padding='max_length', truncation=True)

        model_inputs['labels'] = labels['input_ids']

        item = {key: torch.tensor(val[0]) for key, val in model_inputs.items()}
        return item

    def __len__(self):
        return len(self.inputs)
