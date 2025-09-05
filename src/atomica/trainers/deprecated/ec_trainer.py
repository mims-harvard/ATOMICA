#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer
import torch.nn.functional as F
import numpy as np
import pdb


class ECTrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        self.max_epoch = config.max_epoch
        self.min_lr = config.final_lr
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        return optimizer

    def get_scheduler(self, optimizer):
        if self.config.scheduler == 'exp':
            log_alpha = self.log_alpha
            lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return {
                'scheduler': scheduler,
                'frequency': 'batch'
            }
        elif self.config.scheduler == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epoch, eta_min = self.min_lr)
            return {
                'scheduler': lr_scheduler,
                'frequency': 'epoch'
            }    
        else:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.config.factor,
                patience=self.config.lr_patience,
                min_lr=self.min_lr,
            )  
            return {
                'scheduler': lr_scheduler,
                'frequency': 'val_epoch'
            }     

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)


    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        pred_class = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            atom_positions=batch['atom_positions'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'])
        
        loss = F.binary_cross_entropy(pred_class, label)

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss