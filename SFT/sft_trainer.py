import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
from transformers import AutoTokenizer
import csv
import re
import jsonlines

torch.set_printoptions(threshold=float('inf'))

class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)

class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out

def preprocess_dataset_safety_alignment(train_data_path, eval_data_path, tokenizer, max_length, test_split=0.2):
    preprocessed_data_train = []
    preprocessed_data_eval = []
    lines = jsonlines.open(train_data_path, mode='r')
    for idx, line in enumerate(lines):
        if idx == 0:
            continue

        input = line['input']
        output = line['output']
        tokenized_input = tokenizer(
            input, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)

        tokenized_output = tokenizer(
            output, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)

        positions = (tokenized_input == 126336).nonzero(as_tuple=True)[0]
        preprocessed_data_train.append(
            {
                "input_ids": tokenized_input,
                "output_ids": tokenized_output,
                "positions": positions
            }
        )

    random.shuffle(preprocessed_data_train)

    lines = jsonlines.open(eval_data_path, mode='r')
    for idx, line in enumerate(lines):
        if idx == 0:
            continue

        input = line['input']
        output = line['output']
        tokenized_input = tokenizer(
            input, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)

        tokenized_output = tokenizer(
            output, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length"
        ).input_ids.squeeze(0)

        positions = (tokenized_input == 126336).nonzero(as_tuple=True)[0]
        preprocessed_data_eval.append(
            {
                "input_ids": tokenized_input,
                "output_ids": tokenized_output,
                "positions": positions
            }
        )

    random.shuffle(preprocessed_data_train)
    random.shuffle(preprocessed_data_eval)

    test_data = preprocessed_data_eval
    train_data = preprocessed_data_train
    return train_data, test_data


class dLLMDataCollator_safety_alignment(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["output_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["output_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0

        if "positions" in batch:
            positions = batch.pop("positions")
            all_indices = torch.arange(batch["labels"][0].size(0))
            not_positions = torch.tensor([i for i in all_indices if i not in positions[0]])

            batch["labels"][0][not_positions] = -100
            batch["num_prompt_tokens"] = len(not_positions)

        batch["output_ids"] = noisy_batch.long()

        batch["input_ids"] = batch["output_ids"].clone()
        batch.pop("output_ids")

        return batch


def mask_to_special_token(text, special_token="<|reserved_token_0|>"):
    masks = re.findall(r"(<mask:(\d+)>)", text)
    for mask_tuple in masks:
        mask = mask_tuple[0]
        number = int(mask_tuple[1])
        text = text.replace(mask, special_token*number)

    return text



