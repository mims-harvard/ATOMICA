#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer, LearningRateWarmup
from utils.logger import print_log
from tqdm import tqdm
import numpy as np
import wandb
from scipy.stats import spearmanr
import os
from collections import defaultdict
from torch.optim.lr_scheduler import SequentialLR, LinearLR, LambdaLR

class AffinityTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=1e-3)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        loss, _ = self.share_step(batch, batch_idx, val=False)
        return loss

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _before_train_epoch_start(self):
        # reform batch, with new random batches
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss, pred = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
        )

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            lr = lr[0]
            self.log('lr', lr, batch_idx, val)

        return loss, pred

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
        label_arr = []
        pred_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                label_arr.append(batch['label'].cpu().numpy())
                batch = self.to_device(batch, device)
                metric, pred = self.valid_step(batch, self.valid_global_step)
                pred_arr.append(pred.cpu().numpy())
                metric_arr.append(metric.cpu().item())
                self.valid_global_step += 1
        self.model.train()
        # judge
        pred_arr = np.concatenate(pred_arr)
        label_arr = np.concatenate(label_arr)
        valid_metric = np.sqrt(np.mean(np.square(pred_arr - label_arr))) 
        if self.use_wandb and self._is_main_proc():
            wandb.log({
                'val_loss': np.mean(metric_arr),
                'val_RMSELoss': valid_metric,
                'val_pearson': np.corrcoef(pred_arr, label_arr)[0, 1],
                'val_spearman': spearmanr(pred_arr, label_arr).statistic,
            }, step=self.global_step)
        if self.use_raytune:
            from ray import train as ray_train
            ray_train.report({'val_RMSELoss': float(valid_metric), "epoch": self.epoch})
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            torch.save(module_to_save, save_path)
            self._maintain_topk_checkpoint(valid_metric, save_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if self.epoch < self.config.warmup_epochs or self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        print_log(f"Patience: {self.patience}")
        self.last_valid_metric = valid_metric
        if self.epoch > self.config.warmup_epochs:
            self.best_valid_metric = min(self.best_valid_metric, valid_metric) if self.config.metric_min_better else max(self.best_valid_metric, valid_metric)
        # write valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}


class AffinityNoisyNodesTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr if self.config.warmup_steps == 0 else self.config.warmup_start_lr, weight_decay=1e-3)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        if self.config.warmup_steps > 0:
            scheduler = LearningRateWarmup(optimizer, self.config.warmup_steps, self.config.warmup_start_lr, self.config.warmup_end_lr, scheduler)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }
        
    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        loss_pred, loss_denoise, pred = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            receptor_segment=batch['noisy_segment'], 
            atom_score=batch['atom_score'], 
            atom_eps=batch['atom_eps'], 
            tr_score=batch['tr_score'], 
            tr_eps=batch['tr_eps'],
            rot_score=batch['rot_score'],
            tor_score=batch['tor_score'],
            tor_edges=batch['tor_edges'],
            tor_batch=batch['tor_batch'],
        )

        self.log(f'Loss/Train', loss_pred+loss_denoise, batch_idx)

        lr = self.optimizer.param_groups[0]['lr']
        self.log('lr', lr, batch_idx)

        return loss_pred, loss_denoise

    def valid_step(self, batch, batch_idx):
        return self.model.infer(batch)

    def _before_train_epoch_start(self):
        # reform batch, with new random batches
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def _train_epoch(self, device):
        self._before_train_epoch_start()
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            if self.resume_index > 0:
                self.train_loader.sampler.set_epoch(epoch=self.epoch, resume_index=self.resume_index)
                print_log(f"Resume training from epoch {self.epoch}, global step {self.global_step}")
                self.resume_index = 0
            else:
                self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training") if self._is_main_proc() else enumerate(self.train_loader)
        metric_dict = defaultdict(list)
        print(f"NUMBATCHES = {len(self.train_loader)}")
        for batch_idx, batch in t_iter:
            try:
                batch = self.to_device(batch, device)
                loss_pred, loss_denoise = self.train_step(batch, self.global_step)
                loss = loss_pred + loss_denoise
                self.optimizer.zero_grad()
                loss.backward()
                metric_dict["loss"].append(loss.detach().cpu().item())
                metric_dict["pred_loss"].append(loss_pred.detach().cpu().item())
                metric_dict["denoise_loss"].append(loss_denoise.detach().cpu().item())
                if self.use_wandb and self._is_main_proc():
                    wandb.log({'train_loss': loss.detach().cpu().item(),
                               'train_MSELoss': loss_pred.detach().cpu().item(),
                               'train_RMSELoss': np.sqrt(loss_pred.detach().cpu().item()),
                               'train_denoise_loss': loss_denoise.detach().cpu().item()}, 
                               step=self.global_step)
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                if hasattr(t_iter, 'set_postfix'):
                    t_iter.set_postfix(loss=loss.detach().cpu().item(), version=self.version)
                self.global_step += 1
                if self.sched_freq == 'batch':
                    if isinstance(self.scheduler, LearningRateWarmup):
                        self.scheduler.step(self.global_step)
                    else:
                        self.scheduler.step()
                if self.use_wandb and self._is_main_proc():
                    wandb.log({f'lr': self.optimizer.param_groups[-1]['lr']}, step=self.global_step)
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
            wandb.log({
                'train_epoch_MSELoss': np.mean(metric_dict["pred_loss"]),
                'train_epoch_denoise_loss': np.mean(metric_dict["denoise_loss"]),
                'train_epoch_loss': np.mean(metric_dict["loss"]),
            }, step=self.global_step)
        if self.sched_freq == 'epoch':
            if isinstance(self.scheduler, LearningRateWarmup):
                self.scheduler.step(self.epoch)
            else:
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

        label_arr = []
        pred_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader, total=len(self.valid_loader), desc="Validation") if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                label_arr.append(batch["label"].cpu().numpy())
                batch = self.to_device(batch, device)
                pred = self.valid_step(batch, self.valid_global_step)
                pred_arr.append(pred.cpu().numpy())
                self.valid_global_step += 1
        self.model.train()
        # judge
        pred_arr = np.concatenate(pred_arr)
        label_arr = np.concatenate(label_arr)
        valid_metric = np.sqrt(np.mean(np.square(pred_arr - label_arr)))
        if self.use_wandb and self._is_main_proc():
            wandb.log({
                'val_RMSELoss': valid_metric,
                'val_pearson': np.corrcoef(pred_arr, label_arr)[0, 1],
                'val_spearman': spearmanr(pred_arr, label_arr).statistic,
            }, step=self.global_step)
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
