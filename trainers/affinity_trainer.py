#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import exp, pi, cos, log
import torch
from .abs_trainer import Trainer


class AffinityTrainer(Trainer):

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
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'])

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            lr = lr[0]
            self.log('lr', lr, batch_idx, val)

        return loss
