import os
import sys
import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import torch
import transformers
from transformers import TrainingArguments
import wandb

from arguments import get_arguments

from datasets_ours.get_dataset import get_dataset
from datasets_ours.text_collator import TextCollator
from models.get_model import get_model
from text_trainer import TextTrainer, StopAfterEpoch
from utils.compute_metrics import get_metric

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(42)

    model, tokenizer = get_model(args)
    train_dataset, val_dataset = get_dataset(args, tokenizer)

    if args.local_rank==0 and not (args.is_submission or args.split_valid_by_paragraph):
        wandb.init(project='2025_SW_Competition', name=f'{args.save_dir}')

    training_args = TrainingArguments(
        output_dir=f"./ckpt/{args.save_dir}",
        evaluation_strategy = 'epoch',
        #eval_strategy='epoch',
        #eval_steps=args.eval_steps,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        metric_for_best_model="roc_auc",
        save_strategy="epoch",
        save_total_limit=10,
        #save_steps=args.save_steps,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        report_to="wandb",
        dataloader_num_workers=16,
        save_safetensors=True,        # ⇒ write .safetensors instead of .bin
        save_only_model=True, #remove optimizer, scheduler, etc.
    )

    #num_epoch = 10 #2 if args.is_kfold else 5
    #print(f"[Warning] Training for {num_epoch} epochs")
        
    trainer = TextTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=TextCollator(args, tokenizer),
        compute_metrics=get_metric(args),
        args_original=args,
        #callbacks=[StopAfterEpoch(num_epoch)]
    )

    if not (args.is_submission or args.split_valid_by_paragraph):
       print(f"Training {args.save_dir} model...")
       trainer.train()
    else:
        metric = trainer.evaluate()
        print(f"Evaluation results: {metric}")


if __name__=="__main__":

    args = get_arguments()
    main(args)