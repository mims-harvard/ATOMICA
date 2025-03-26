from math import log
import torch
from .abs_trainer import Trainer
from utils.logger import print_log
from tqdm import tqdm
import numpy as np
import wandb
import os
import json

class ProtInterfaceTrainer(Trainer):

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.config.cycle_steps, eta_min=self.config.final_lr)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def lr_weight(self, step):
        if self.global_step >= self.config.warmup:
            return 0.99 ** self.epoch
        return (self.global_step + 1) * 1.0 / self.config.warmup

    def train_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=False)
        return loss

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _before_train_epoch_start(self):
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(
            batch_cmplx=batch['cmplx'],
            batch_prot=batch['prot'],
        )

        log_type = 'Validation' if val else 'Train'

        self.log(f'Loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            lr = lr[0]
            self.log('lr', lr, batch_idx, val)

        return loss

    def _valid_epoch(self, device):
        if self.valid_loader is None:
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
                config_path = os.path.join(self.model_dir, 'config.json')
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                if self.config.save_topk < 0 or (self.config.max_epoch - self.epoch <= self.config.save_topk):
                    print_log(f'No validation, save path: {save_path}')
                    torch.save(module_to_save, save_path)
                    torch.save(module_to_save.state_dict(), weights_path)
                    with open(config_path, 'w') as fout:
                        json.dump(module_to_save.get_config(), fout, indent=4)
                else:
                    print_log('No validation')
            return

        metric_arr = []
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                metric = self.valid_step(batch, self.valid_global_step)
                metric_arr.append(metric)
                self.valid_global_step += 1
        self.model.train()
        valid_metric = np.mean(metric_arr)
        if self.use_wandb and self._is_main_proc():
            wandb.log({
                'val_loss': valid_metric,
            }, step=self.global_step)
        if self.use_raytune:
            from ray import train as ray_train
            ray_train.report({'val_RMSELoss': float(valid_metric), "epoch": self.epoch})
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
            weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
            config_path = os.path.join(self.model_dir, 'config.json')
            torch.save(module_to_save, save_path)
            torch.save(module_to_save.state_dict(), weights_path)
            with open(config_path, 'w') as fout:
                json.dump(module_to_save.get_config(), fout, indent=4)
            self._maintain_topk_checkpoint(valid_metric, save_path)
            self._maintain_topk_weights(valid_metric, weights_path)
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if self.epoch < self.config.warmup_epochs or self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        print_log(f"Patience: {self.patience}")
        self.last_valid_metric = valid_metric
        if self.epoch > self.config.warmup_epochs:
            self.best_valid_metric = min(self.best_valid_metric, valid_metric) if self.config.metric_min_better else max(self.best_valid_metric, valid_metric)
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}