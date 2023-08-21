"""
The cross validation function for finetuning.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/train/cross_validate.py
"""
import os
import time
from argparse import Namespace
from logging import Logger
from typing import Tuple

import numpy as np
import pandas as pd

from grover.util.utils import get_task_names
from grover.util.utils import makedirs
from task.run_evaluation import run_evaluation
from task.train import run_training
from datetime import datetime

import torch.distributed as dist
import torch.multiprocessing as mp

def cross_validate(args: Namespace, logger: Logger = None) -> Tuple[float, float]:
    """
    k-fold cross validation.

    :return: A tuple of mean_score and std_score.
    """
    info = logger.info if logger is not None else print

    mean_score=0.0
    std_score=0.0
    
    # Initialize relevant variables
    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    all_scores = []

    # Run training with different random seeds for each fold on rank 0
    info('hyperparameter')
    info(f'batch_size {args.batch_size}')
    info(f'init_lr {args.init_lr}')
    info(f'max_lr {args.max_lr}')
    info(f'final_lr {args.final_lr}')
    info(f'dropout {args.dropout}')
    info(f'attn_hidden {args.attn_hidden}')
    info(f'attn_out {args.attn_out}')
    info(f'dist_coff {args.dist_coff}')
    info(f'bond_drop_rate {args.bond_drop_rate}')
    info(f'ffn_num_layers {args.ffn_num_layers}')
    info(f'ffn_hidden_size {args.ffn_hidden_size}')
    st=datetime.now()
    time_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    # info(f'HP_start time: {time_start}')
    # PyTorch DDP
    world_size=args.num_procs
    rank = int(os.environ["LOCAL_RANK"])

    for fold_num in range(args.num_folds):
        fold_start = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        info(f'Fold {fold_num}')
        info(f'start time: {fold_start}')
        args.seed = init_seed + fold_num
        args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
        makedirs(args.save_dir)
        info(f'save_dir {args.save_dir}')
        if args.parser_name == "finetune":
            model_scores = run_training(args, time_start, logger)
        else:
            model_scores = run_evaluation(args, logger)
        # PyTorch DDP
        if rank == 0:
            all_scores.append(model_scores)
    if rank == 0:
        all_scores=np.array(all_scores)
        time_end = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        et=datetime.now()
        diff=et-st
        info(f'HP_end time: {time_end}')
        info(f'total time: {diff}')

        # PyTorch DDP
        #  dist.destroy_process_group()

        # all_scores = np.array(all_scores)
        # Report scores for each fold

        info(f'{args.num_folds}-fold cross validation')
        for fold_num, scores in enumerate(all_scores):
            info(f'Seed {init_seed + fold_num} ==> test {args.metric} = {np.nanmean(scores):.6f}')

            if args.show_individual_scores:
                for task_name, score in zip(task_names, scores):
                    info(f'Seed {init_seed + fold_num} ==> test {task_name} {args.metric} = {score:.6f}')

        # Report scores across models
        avg_scores = np.nanmean(all_scores)  # average score for each model across tasks
        mean_score, std_score = np.nanmean(avg_scores), np.nanstd(avg_scores)
        info(f'overall_{args.split_type}_test_{args.metric}={mean_score:.6f}')
        info(f'std={std_score:.6f}')

        if args.show_individual_scores:
            for task_num, task_name in enumerate(task_names):
                 info(f'Overall test {task_name} {args.metric} = '
                     f'{np.nanmean(all_scores[:, task_num]):.6f} +/- {np.nanstd(all_scores[:, task_num]):.6f}')
    return mean_score, std_score
