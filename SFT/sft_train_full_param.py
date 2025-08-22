import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
# from sft_trainer import *
import torch.distributed as dist
import random
import numpy as np
from accelerate import Accelerator
from accelerate.utils import DummyScheduler, DummyOptim
from sft_trainer_xu import dLLMTrainer, dLLMSFTDataset, dLLMDataCollator_safety_alignment, preprocess_dataset_safety_alignment

def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Initialize argument parser
def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument(
        "--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Name of the pretrained model"
    )
    parser.add_argument("--local_batch_size", type=int, default=1, help="Local batch size per GPU")
    parser.add_argument(
        "--max_length", type=int, default=4096, help="Maximum sequence length for tokenization"
    )
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=32, help="Gradient accumulation steps for global batch size 256")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sft_save",
        help="Directory to save model checkpoints and logs",
    )
    parser.add_argument("--job_name", type=str, default="llada-s1", help="Job Name")
    parser.add_argument("--train_data", type=str, default="simplescaling/s1K", help="Path to training data")
    parser.add_argument("--eval_data", type=str, default="simplescaling/s1K", help="Path to training data")
    parser.add_argument(
        "--debugging", action="store_true", help="Use while debugging model - only disables wandb logging"
    )

    return parser.parse_args()


# Model loading for full parameter fine-tuning
def load_model_and_tokenizer(args):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="right", trust_remote_code=True, use_fast=True
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    return tokenizer, model

def load_data_safety_alignment(args, tokenizer):
    train_data, eval_data = preprocess_dataset_safety_alignment(args.train_data, args.eval_data, tokenizer, args.max_length)
    print("Train data length: ", len(train_data))
    print("Eval data length: ", len(eval_data))
    train_dataset = dLLMSFTDataset(train_data, tokenizer, args.max_length)
    eval_dataset = dLLMSFTDataset(eval_data, tokenizer, args.max_length, eval=True)
    return train_dataset, eval_dataset

# Training setup
def train_model(args, tokenizer, model):
    # Load dataset
    train_dataset, eval_dataset = load_data_safety_alignment(args, tokenizer)

    # Initialize accelerator
    accelerator = Accelerator()
    print("total_dataset number",len(train_dataset))
    # Calculate total training steps
    num_training_steps = len(train_dataset) * args.num_epochs // (args.local_batch_size * args.grad_accum_steps)

    # Training arguments setup with DeepSpeed
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.job_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.local_batch_size,
        per_device_eval_batch_size=args.local_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="steps",
        save_strategy="epoch",
        eval_steps=500,
        logging_steps=25,
        # save_steps=300,
        save_total_limit=20,
        learning_rate=2.5e-5,  # 初始学习率会被scheduler覆盖
        # load_best_model_at_end=True,
        weight_decay=0.1,
        max_grad_norm=1.0,
        bf16=True,
        remove_unused_columns=False,
        save_only_model=True,
        report_to="none"
    )

    # Initialize Trainer with custom dLLMTrainer
    trainer = dLLMTrainer(
        model=model,
        args=training_args,
        # data_collator=dLLMDataCollator(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        data_collator=dLLMDataCollator_safety_alignment(tokenizer=tokenizer, mask_token_id=126336, max_length=args.max_length),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    init_seed(42)
    # Parse command-line arguments
    args = parse_args()

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args)

    # Train the model
    train_model(args, tokenizer, model)


"""
accelerate launch --main_process_port 29501 --config_file full_param_accelerate_config.yaml sft_train_full_param.py \
--model_name /data/models/LLaDA-8B-Instruct \
--max_length 4096 \
--num_epochs 5 \
--grad_accum_steps 1 \
--job_name LLaDA_run_test \
--train_data /data/safety/alpaca+wildguardmix/mix_5000/train.jsonl \
--eval_data /data/safety/alpaca+wildguardmix/mix_5000/test.jsonl \
--local_batch_size 1 \
--output_dir alpaca+wildguardmix_mix_5000


"""

