from math import exp, log
import torch
from .abs_trainer import Trainer
from utils.logger import print_log
from tqdm import tqdm
import numpy as np
import wandb
from scipy.stats import spearmanr
import os
import json
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

class AffinityTrainer(Trainer):

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
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def share_step(self, batch, batch_idx, val=False):
        loss, pred = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None),
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
            weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
            config_path = os.path.join(self.model_dir, 'config.json')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
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

class ClassifierTrainer(Trainer):

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
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def share_step(self, batch, batch_idx, val=False):
        loss, pred = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None),
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
        valid_metric = np.mean(metric_arr)
        label_arr = np.concatenate(label_arr)
        pred_arr = np.concatenate(pred_arr)
        auroc = roc_auc_score(label_arr, pred_arr)
        precision, recall, _ = precision_recall_curve(label_arr, pred_arr)
        auprc = auc(recall, precision)
        freq_baseline = np.mean(label_arr)

        if self.use_wandb and self._is_main_proc():
            wandb.log({
                'val_loss': valid_metric,
                'val_auroc': auroc,
                'val_auprc': auprc,
                'val_delta_auprc': auprc - freq_baseline,
            }, step=self.global_step)
        if self.use_raytune:
            from ray import train as ray_train
            ray_train.report({'val_loss': float(valid_metric), "epoch": self.epoch})
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
            config_path = os.path.join(self.model_dir, 'config.json')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
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


class MultiClassClassifierTrainer(Trainer):

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
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def share_step(self, batch, batch_idx, val=False):
        loss, pred = self.model(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            label=batch['label'],
            block_embeddings=batch.get('block_embeddings', None),
            block_embeddings0=batch.get('block_embeddings0', None),
            block_embeddings1=batch.get('block_embeddings1', None),
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
        valid_metric = np.mean(metric_arr)
        label_arr = np.concatenate(label_arr)
        pred_arr = np.concatenate(pred_arr)

        frequency_baseline = np.bincount(label_arr) / len(label_arr)
        auprc_per_class = []
        for i in range(self.model.num_classes):
            if len(label_arr==i) == 0:
                continue
            precision, recall, _ = precision_recall_curve(label_arr == i, pred_arr[:, i])
            auprc = auc(recall, precision)
            auprc_per_class.append(auprc)
        
        mean_auprc = np.mean(auprc_per_class)
        mean_delta_auprc = mean_auprc - np.mean(frequency_baseline)

        if self.use_wandb and self._is_main_proc():
            wandb.log({
                'val_loss': valid_metric,
                'val_auprc': mean_auprc,
                'val_delta_auprc': mean_delta_auprc,
            }, step=self.global_step)
        if self.use_raytune:
            from ray import train as ray_train
            ray_train.report({'val_RMSELoss': float(valid_metric), "epoch": self.epoch})
        if self._is_main_proc():
            save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
            weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
            config_path = os.path.join(self.model_dir, 'config.json')
            module_to_save = self.model.module if self.local_rank == 0 else self.model
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
