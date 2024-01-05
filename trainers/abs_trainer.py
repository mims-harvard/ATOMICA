#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.logger import print_log

########### Import your packages below ##########
import wandb


class TrainConfig:
    def __init__(self, save_dir, lr, max_epoch, warmup=0,
                 metric_min_better=True, patience=3,
                 grad_clip=None, save_topk=-1,  # -1 for save all
                 **kwargs):
        self.save_dir = save_dir
        self.lr = lr
        self.max_epoch = max_epoch
        self.warmup = warmup
        self.metric_min_better = metric_min_better
        self.patience = patience if patience > 0 else max_epoch
        self.grad_clip = grad_clip
        self.save_topk = save_topk
        self.__dict__.update(kwargs)

    def add_parameter(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__class__) + ': ' + str(self.__dict__)


class Trainer:
    def __init__(self, model, train_loader, valid_loader, config):
        self.model = model
        self.config = config
        self.optimizer = self.get_optimizer()
        sched_config = self.get_scheduler(self.optimizer)
        if sched_config is None:
            sched_config = {
                'scheduler': None,
                'frequency': None
            }
        self.scheduler = sched_config['scheduler']
        self.sched_freq = sched_config['frequency']
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # distributed training
        self.local_rank = -1

        # log
        self.version = self._get_version()
        self.config.save_dir = os.path.join(self.config.save_dir, f'version_{self.version}')
        self.model_dir = os.path.join(self.config.save_dir, 'checkpoint')
        self.writer = None  # initialize right before training
        self.writer_buffer = {}
        self.use_wandb = False

        # training process recording
        self.valid_requires_grad = False
        self.global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.last_valid_metric = None
        self.topk_ckpt_map = []  # smaller index means better ckpt
        self.patience = self.config.patience

    @classmethod
    def to_device(cls, data, device):
        if isinstance(data, dict):
            for key in data:
                data[key] = cls.to_device(data[key], device)
        elif isinstance(data, list) or isinstance(data, tuple):
            res = [cls.to_device(item, device) for item in data]
            data = type(data)(res)
        elif hasattr(data, 'to'):
            data = data.to(device)
        return data
    
    def set_valid_requires_grad(self, require=False):
        self.valid_requires_grad = require

    def _is_main_proc(self):
        return self.local_rank == 0 or self.local_rank == -1

    def _get_version(self):
        version, pattern = -1, r'version_(\d+)'
        if os.path.exists(self.config.save_dir):
            for fname in os.listdir(self.config.save_dir):
                ver = re.findall(pattern, fname)
                if len(ver):
                    version = max(int(ver[0]), version)
        return version + 1

    def _before_train_epoch_start(self):
        return

    def _train_epoch(self, device):
        self._before_train_epoch_start()
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(self.train_loader) if self._is_main_proc() else self.train_loader
        for batch in t_iter:
            batch = self.to_device(batch, device)
            loss = self.train_step(batch, self.global_step)
            self.optimizer.zero_grad()
            loss.backward()

            if self.use_wandb:
                wandb.log({f'train_MSELoss': loss.item()}, step=self.global_step)
                wandb.log({f'train_RMSELoss': np.sqrt(loss.item())}, step=self.global_step)

            # for name, param in self.model.named_parameters():
            #     if not param.requires_grad:
            #         continue
            #     print(name)
            #     print(torch.norm(param.grad))
            #     # print(param.grad[0])

            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            if hasattr(t_iter, 'set_postfix'):
                t_iter.set_postfix(loss=loss.item(), version=self.version)
            self.global_step += 1
            if self.sched_freq == 'batch':
                self.scheduler.step()
        if self.sched_freq == 'epoch':
            self.scheduler.step()
    
    def _valid_epoch(self, device):
        if self.valid_loader is None:
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                if self.config.save_topk < 0 or (self.config.max_epoch - self.epoch <= self.config.save_topk):
                    print_log(f'No validation, save path: {save_path}')
                    torch.save(module_to_save, save_path)
                else:
                    print_log('No validation')
            return

        metric_arr = []
        self.model.eval()
        with torch.set_grad_enabled(self.valid_requires_grad):
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        valid_metric = np.mean(metric_arr)
        if self.use_wandb:
            wandb.log({f'val_MSELoss': valid_metric.item()}, step=self.global_step)
            wandb.log({f'val_RMSELoss': np.sqrt(valid_metric)}, step=self.global_step)
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            torch.save(module_to_save, save_path)
            self._maintain_topk_checkpoint(valid_metric, save_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        self.last_valid_metric = valid_metric
        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}
    
    def _metric_better(self, new):
        old = self.last_valid_metric
        if old is None:
            return True
        if self.config.metric_min_better:
            return new < old
        else:
            return old < new

    def _maintain_topk_checkpoint(self, valid_metric, ckpt_path):
        topk = self.config.save_topk
        if self.config.metric_min_better:
            better = lambda a, b: a < b
        else:
            better = lambda a, b: a > b
        insert_pos = len(self.topk_ckpt_map)
        for i, (metric, _) in enumerate(self.topk_ckpt_map):
            if better(valid_metric, metric):
                insert_pos = i
                break
        self.topk_ckpt_map.insert(insert_pos, (valid_metric, ckpt_path))

        # maintain topk
        if topk > 0:
            while len(self.topk_ckpt_map) > topk:
                last_ckpt_path = self.topk_ckpt_map[-1][1]
                os.remove(last_ckpt_path)
                self.topk_ckpt_map.pop()

        # save map
        topk_map_path = os.path.join(self.model_dir, 'topk_map.txt')
        with open(topk_map_path, 'w') as fout:
            for metric, path in self.topk_ckpt_map:
                fout.write(f'{metric}: {path}\n')

    def train(self, device_ids, local_rank, use_wandb=False):
        self.use_wandb = use_wandb
        # set local rank
        self.local_rank = local_rank
        # init writer
        if self._is_main_proc():
            self.writer = SummaryWriter(self.config.save_dir)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            with open(os.path.join(self.config.save_dir, 'train_config.json'), 'w') as fout:
                json.dump(self.config.__dict__, fout)
        # main device
        main_device_id = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device('cpu' if main_device_id == -1 else f'cuda:{main_device_id}')
        self.model.to(device)
        if local_rank != -1:
            print_log(f'Using data parallel, local rank {local_rank}, all {device_ids}')
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            print_log(f'training on {device_ids}')
        for _ in range(self.config.max_epoch):
            print_log(f'epoch{self.epoch} starts') if self._is_main_proc() else 1
            self._train_epoch(device)
            print_log(f'validating ...') if self._is_main_proc() else 1
            self._valid_epoch(device)
            self.epoch += 1
            if self.patience <= 0:
                break

    def log(self, name, value, step, val=False):
        if self._is_main_proc():
            if isinstance(value, torch.Tensor):
                value = value.cpu().item()
            if val:
                if name not in self.writer_buffer:
                    self.writer_buffer[name] = []
                self.writer_buffer[name].append(value)
            else:
                self.writer.add_scalar(name, value, step)

    ########## Overload these functions below ##########
    # define optimizer
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    # scheduler example: linear. Return None if no scheduler is needed.
    def get_scheduler(self, optimizer):
        lam = lambda epoch: 1 / (epoch + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
        return {
            'scheduler': scheduler,
            'frequency': 'epoch' # or batch
        }

    # train step, note that batch should be dict/list/tuple/instance. Objects with .to(device) attribute will be automatically moved to the same device as the model
    def train_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/train', loss, batch_idx)
        return loss

    # validation step
    def valid_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log('Loss/validation', loss, batch_idx, val=True)
        return loss
