#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer
from utils.logger import print_log
from torch.utils.tensorboard import SummaryWriter
import os
import json
from tqdm import tqdm
import wandb
import numpy as np
from collections import defaultdict

class PretrainTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-3)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50000, eta_min=self.config.final_lr)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _before_train_epoch_start(self):
        # reform batch, with new random batches
        # self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        try:
            loss = self.model(
                Z=batch['X'], B=batch['B'], A=batch['A'],
                atom_positions=batch['atom_positions'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                label=None,
                return_loss=True)

            log_type = 'Validation' if val else 'Train'

            self.log(f'Loss/loss/{log_type}', loss.loss, batch_idx, val)
            self.log(f'Loss/noise_loss/{log_type}', loss.noise_loss, batch_idx, val)
            self.log(f'Loss/noise_level_loss/{log_type}', loss.noise_level_loss, batch_idx, val)
            self.log(f'Loss/align_loss/{log_type}', loss.align_loss, batch_idx, val)

            if not val:
                lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
                lr = lr[0]
                self.log('lr', lr, batch_idx, val)

            return loss
        except RuntimeError as e:
            if "out of memory" in str(e) and torch.cuda.is_available():
                print_log(e, level='ERROR')
                print_log(
                    f"""Out of memory error, skipping batch {batch_idx}, num_nodes={batch['X'].shape[0]}, 
                    num_blocks={batch['B'].shape[0]}, batch_size={batch['lengths'].shape[0]},
                    max_item_block_size={batch['lengths'].max()}""", level='ERROR')
                for p in self.model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                return None
            else:
                raise e

    def _train_epoch(self, device, validation_freq=5000):
        self._before_train_epoch_start()
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(enumerate(self.train_loader)) if self._is_main_proc() else enumerate(self.train_loader)
        metric_dict = defaultdict(list)
        print(f"NUMBATCHES = {len(self.train_loader)}")
        for batch_idx, batch in t_iter:
            try:
                batch = self.to_device(batch, device)
                loss_obj = self.train_step(batch, self.global_step)
                if loss_obj is None:
                    continue # Out of memory
                self.optimizer.zero_grad()
                loss_obj.loss.backward()
                metric_dict["loss"].append(loss_obj.loss.cpu().item())
                metric_dict["atom_loss"].append(loss_obj.atom_loss.cpu().item())
                metric_dict["translation_loss"].append(loss_obj.translation_loss.cpu().item())
                metric_dict["rotation_loss"].append(loss_obj.rotation_loss.cpu().item())
                if self.use_wandb and self._is_main_proc():
                    wandb.log({f'train_MSELoss': loss_obj.loss.item()}, step=self.global_step)
                    wandb.log({f'train_RMSELoss': np.sqrt(loss_obj.loss.item())}, step=self.global_step)
                    wandb.log({f'train_atom_loss': loss_obj.atom_loss}, step=self.global_step)
                    wandb.log({f'train_translation_loss': loss_obj.translation_loss}, step=self.global_step)
                    wandb.log({f'train_rotation_loss': loss_obj.rotation_loss}, step=self.global_step)
                    wandb.log({f'train_translation_base': loss_obj.translation_base}, step=self.global_step)
                    wandb.log({f'train_rotation_base': loss_obj.rotation_base}, step=self.global_step)
                    if batch_idx % 500 == 0 and batch_idx > 0:
                        start_idx = max(0, len(metric_dict["loss"]) - 500)
                        wandb.log({f'train_last500_MSELoss': np.mean(metric_dict["loss"][start_idx:])}, step=self.global_step)
                        wandb.log({f'train_last500_atom_loss': np.mean(metric_dict["atom_loss"][start_idx:])}, step=self.global_step)
                        wandb.log({f'train_last500_translation_loss': np.mean(metric_dict["translation_loss"][start_idx:])}, step=self.global_step)
                        wandb.log({f'train_last500_rotation_loss': np.mean(metric_dict["rotation_loss"][start_idx:])}, step=self.global_step)

                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                if hasattr(t_iter, 'set_postfix'):
                    t_iter.set_postfix(loss=loss_obj.loss.item(), version=self.version)
                self.global_step += 1
                if self.sched_freq == 'batch':
                    self.scheduler.step()
                if self.use_wandb and self._is_main_proc():
                    wandb.log({f'lr': self.optimizer.param_groups[-1]['lr']}, step=self.global_step)
                if batch_idx % validation_freq == 0 and batch_idx > 0:
                    print_log(f'validating ...') if self._is_main_proc() else 1
                    self._valid_epoch(device)
                    self._before_train_epoch_start()
            except RuntimeError as e:
                if "out of memory" in str(e) and torch.cuda.is_available():
                    print_log(e, level='ERROR')
                    print_log(
                        f"""Out of memory error, skipping batch {batch_idx}, num_nodes={batch['X'].shape[0]}, 
                        num_blocks={batch['B'].shape[0]}, batch_size={batch['lengths'].shape[0]},
                        max_item_block_size={batch['lengths'].max()}""", level='ERROR'
                    )
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        if self.use_wandb and self._is_main_proc():
            wandb.log({f'train_epoch_MSELoss': np.mean(metric_dict["loss"])}, step=self.global_step)
            wandb.log({f'train_epoch_atom_loss': np.mean(metric_dict["atom_loss"])}, step=self.global_step)
            wandb.log({f'train_epoch_translation_loss': np.mean(metric_dict["translation_loss"])}, step=self.global_step)
            wandb.log({f'train_epoch_rotation_loss': np.mean(metric_dict["rotation_loss"])}, step=self.global_step)
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

        metric_dict = defaultdict(list)
        self.model.eval()
        with torch.set_grad_enabled(self.valid_requires_grad):
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                if metric is None:
                    continue # Out of memory
                metric_dict["loss"].append(metric.loss.cpu().item())
                metric_dict["atom_loss"].append(metric.atom_loss.cpu().item())
                metric_dict["translation_loss"].append(metric.translation_loss.cpu().item())
                metric_dict["rotation_loss"].append(metric.rotation_loss.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        valid_metric = np.mean(metric_dict["loss"])
        if self.use_wandb and self._is_main_proc():
            wandb.log({f'val_MSELoss': valid_metric.item()}, step=self.global_step)
            wandb.log({f'val_RMSELoss': np.sqrt(valid_metric)}, step=self.global_step)
            wandb.log({f'val_atom_loss': np.mean(metric_dict["atom_loss"])}, step=self.global_step)
            wandb.log({f'val_translation_loss': np.mean(metric_dict["translation_loss"])}, step=self.global_step)
            wandb.log({f'val_rotation_loss': np.mean(metric_dict["rotation_loss"])}, step=self.global_step)
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