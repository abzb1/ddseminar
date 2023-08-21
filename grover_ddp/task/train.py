"""
The training function used in the finetuning task.
"""
import csv
import logging
import os
import pickle
import time
from argparse import Namespace
from logging import Logger
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, DistributedSampler

from grover.data import MolCollator
from grover.data import StandardScaler
from grover.util.metrics import get_metric_func
from grover.util.nn_utils import initialize_weights, param_count
from grover.util.scheduler import NoamLR
from grover.util.utils import build_optimizer, build_lr_scheduler, makedirs, load_checkpoint, get_loss_func, \
    save_checkpoint, build_model
from grover.util.utils import get_class_sizes, get_data, split_data, get_task_names
from task.predict import predict, evaluate, evaluate_predictions

from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def print_rank_0(log, rank):
    if rank == 0:
        print(log, flush=True)

def train(epoch, model, data, loss_func, optimizer, scheduler,
          shared_dict, args: Namespace, n_iter: int = 0,
          logger: logging.Logger = None, num_workers: int = 4):
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    # debug = logger.debug if logger is not None else print

    model.train()

    # data.shuffle()

    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0


    mol_collator = MolCollator(shared_dict=shared_dict, args=args)

    num_workers = num_workers

    if type(data) == DataLoader:
        mol_loader = data
    else:
        mol_loader = DataLoader(data, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, collate_fn=mol_collator)

    for _, item in enumerate(mol_loader):
        _, batch, features_batch, mask, targets = item
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch, features_batch)
        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

    return n_iter, cum_loss_sum / cum_iter_count


def run_training(args: Namespace, time_start, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    writer=SummaryWriter()

    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # PyTorch DDP distinit setup
    if args.torchddp:
        # PyTorch DDP
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        debug(f"Start running DDP GROVER  on rank {local_rank}.")
        debug(f'setup rank: {rank} process')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        debug(f'rank: {rank} process setup complete')
        num_workers=int(mp.cpu_count() / world_size)

    # pin GPU to local rank.
    torch.cuda.set_device(local_rank)
    
    features_scaler, scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger)

    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.checkpoint_paths is not None:
            if len(args.checkpoint_paths) == 1:
                cur_model = 0
            else:
                cur_model = model_idx
            model = load_checkpoint(args.checkpoint_paths[cur_model], current_args=args, logger=logger)
        else:
            model = build_model(model_idx=model_idx, args=args)

        if args.fine_tune_coff != 1 and args.checkpoint_paths is not None:
            debug("Fine tune fc layer with different lr")
            initialize_weights(model_idx=model_idx, model=model.ffn, distinct_init=args.distinct_init)

        # Get loss and metric functions
        loss_func = get_loss_func(args, model)

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # PyTorch DDP: ddp model
        ddp_model=DDP(model, device_ids=[local_rank], output_device=rank, find_unused_parameters=True)

        debug(f"Running DDP on rank: {rank}")
        
        # PyTorch DDP
        optimizer = build_optimizer(ddp_model, args)

        # PyTorch DDP Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Bulid data_loader
        shuffle = True
        mol_collator = MolCollator(shared_dict={}, args=args)
        
        # PyTorch DDP
        if args.torchddp:
            train_sampler=DistributedSampler(train_data)
            train_sampler.set_epoch(epoch=args.epochs)
            train_data=DataLoader(train_data, batch_size=args.batch_size, num_workers=num_workers, drop_last=True, sampler=train_sampler, collate_fn=mol_collator)
        else:
            train_data = DataLoader(train_data,
                                    batch_size=args.batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    collate_fn=mol_collator)
        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        min_val_loss = float('inf')
        
        print(f"data length: {len(train_data)}")
        
        for epoch in range(args.epochs):
            s_time = time.time()
            n_iter, train_loss = train(
                epoch=epoch,
                model=ddp_model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                shared_dict=shared_dict,
                logger=logger,
                num_workers=num_workers
            )
            t_time = time.time() - s_time
            s_time = time.time()
            val_scores, val_loss = evaluate(
                model=model,
                data=val_data,
                loss_func=loss_func,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                shared_dict=shared_dict,
                logger=logger,
                args=args
            )
            v_time = time.time() - s_time
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            # Logged after lr step
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()

            if args.show_individual_scores and rank == 0:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
            print('[{:01d}]'.format(rank),'Epoch: {:04d}'.format(epoch),
                  'loss_train: {:.6f}'.format(train_loss),
                  'loss_val: {:.6f}'.format(val_loss),
                  f'{args.metric}_val: {avg_val_score:.4f}',
                  # 'auc_val: {:.4f}'.format(avg_val_score),
                  'cur_lr: {:.5f}'.format(scheduler.get_lr()[-1]),
                  't_time: {:.4f}s'.format(t_time),
                  'v_time: {:.4f}s'.format(v_time))
            
            # PyTorch DDP: save model on rank 0
            if rank == 0: 
                # Save model checkpoint if improved validation score
                if args.select_by_loss:
                    if val_loss < min_val_loss:
                        min_val_loss, best_epoch = val_loss, epoch
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), ddp_model, scaler, features_scaler, args)
                        # PyTorch DDP
                        # torch.save(ddp_model.state_dict(), os.path.join(save_dir, 'model.pt'))
                else:
                    if args.minimize_score and avg_val_score < best_score or \
                            not args.minimize_score and avg_val_score > best_score:
                        best_score, best_epoch = avg_val_score, epoch
                        save_checkpoint(os.path.join(save_dir, 'model.pt'), ddp_model, scaler, features_scaler, args)
                        # PyTorch DDP
                        # torch.save(ddp_model.state_dict(), os.path.join(save_dir, 'model.pt'))

                if epoch - best_epoch > args.early_stop_epoch:
                    break

        ensemble_scores = 0.0

        # PyTorch DDP
        if rank == 0:
            # Evaluate on test set using model with best validation score
            if args.select_by_loss:
                info(f'Model {model_idx} best val loss = {min_val_loss:.6f} on epoch {best_epoch}')
            else:
                info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        
        # PyTorch DDP
        dist.barrier()
        if rank == 0:
            
            # ddp_model.load_state_dict(torch.load(os.path.join(save_dir)), map_location=map_location)
            model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

            test_preds, _ = predict(
                model=ddp_model,
                data=test_data,
                loss_func=loss_func,
                batch_size=args.batch_size,
                logger=logger,
                shared_dict=shared_dict,
                scaler=scaler,
                args=args
            )

            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )
            """
            if os.path.exists(os.path.join(save_dir,'model.pt')):
                os.remove(os.path.join(save_dir,'model.pt'))
            """
            if len(test_preds) != 0:
                sum_test_preds += np.array(test_preds, dtype=float)
        
            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx} test {args.metric} = {avg_test_score:.6f}')
        
            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                   info(f'Model {model_idx} test {task_name} {args.metric} = {test_score:.6f}')

            # Evaluate ensemble on test set
            avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()

            ensemble_scores = evaluate_predictions(
                preds=avg_test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                dataset_type=args.dataset_type,
                logger=logger
            )

            ind = [['preds'] * args.num_tasks + ['targets'] * args.num_tasks, args.task_names * 2]
            ind = pd.MultiIndex.from_tuples(list(zip(*ind)))
            data = np.concatenate([np.array(avg_test_preds), np.array(test_targets)], 1)
            test_result = pd.DataFrame(data, index=test_smiles, columns=ind)
            test_result.to_csv(os.path.join(args.save_dir, 'test_result.csv'))

            # Average ensemble score
            avg_ensemble_test_score = np.nanmean(ensemble_scores)
            info(f'Ensemble test {args.metric} = {avg_ensemble_test_score:.6f}')

            # Individual ensemble scores
            if args.show_individual_scores:
                for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                    info(f'Ensemble test {task_name} {args.metric} = {ensemble_score:.6f}')
    dist.destroy_process_group() 
    
    return ensemble_scores


def load_data(args, debug, logger):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')
    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args,
                             features_path=args.separate_test_features_path, logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args,
                            features_path=args.separate_val_features_path, logger=logger)
    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type,
                                              sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type,
                                             sizes=(0.8, 0.2, 0.0), seed=args.seed, args=args, logger=logger)
    else:
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                     sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)
    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    #if args.save_smiles_splits:
    #    save_splits(args, test_data, train_data, val_data)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    args.train_data_size = len(train_data)
    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)

        val_targets = val_data.targets()
        scaled_val_targets = scaler.transform(val_targets).tolist()
        val_data.set_targets(scaled_val_targets)
    else:
        scaler = None
    return features_scaler, scaler, shared_dict, test_data, train_data, val_data


def save_splits(args, test_data, train_data, val_data):
    """
    Save the splits.
    :param args:
    :param test_data:
    :param train_data:
    :param val_data:
    :return:
    """
    with open(args.data_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        lines_by_smiles = {}
        indices_by_smiles = {}
        for i, line in enumerate(reader):
            smiles = line[0]
            lines_by_smiles[smiles] = line
            indices_by_smiles[smiles] = i

    all_split_indices = []
    for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
        with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles'])
            for smiles in dataset.smiles():
                writer.writerow([smiles])
        with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for smiles in dataset.smiles():
                writer.writerow(lines_by_smiles[smiles])
        split_indices = []
        for smiles in dataset.smiles():
            split_indices.append(indices_by_smiles[smiles])
            split_indices = sorted(split_indices)
        all_split_indices.append(split_indices)
    with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
        pickle.dump(all_split_indices, f)
    return writer