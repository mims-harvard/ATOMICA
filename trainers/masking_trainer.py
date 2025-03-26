import torch
from .abs_trainer import Trainer
from utils.logger import print_log
import os
from tqdm import tqdm
import wandb
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score
import json

class MaskingTrainer(Trainer):
    def __init__(self, model, train_loader, valid_loader, config, resume_state=None):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        super().__init__(model, train_loader, valid_loader, config)
        self.training_state_dir = os.path.join(self.config.save_dir, 'training_state')
        if resume_state is not None:
            if not torch.distributed.is_available():
                raise NotImplementedError("Only DistributedDataParallel supports resuming training from a specific batch.")
            self.global_step = resume_state['global_step']
            self.epoch = resume_state['epoch']
            self.optimizer.load_state_dict(resume_state['optimizer'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
            self.scheduler.load_state_dict(resume_state['scheduler'])
            self.resume_index = resume_state['resume_index'] % (len(self.train_loader) * self.train_loader.batch_size)
        else:
            self.resume_index = 0                
            
    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=1e-3)
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
        loss, _ = self.share_step(batch, batch_idx, val=False)
        return loss

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)

    def _before_train_epoch_start(self):
        # reform batch, with new random batches
        self.train_loader.dataset._form_batch()
        return super()._before_train_epoch_start()

    def share_step(self, batch, batch_idx, val=False):
        try:
            loss, logits = self.model(
                Z=batch['X'], B=batch['B'], A=batch['A'],
                block_lengths=batch['block_lengths'],
                lengths=batch['lengths'],
                segment_ids=batch['segment_ids'],
                masked_blocks=batch['masked_blocks'],
                masked_labels=batch['masked_labels'],
            )

            if not val:
                lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
                lr = lr[0]
                self.log('lr', lr, batch_idx, val)

            return loss, logits
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
                return None, None
            else:
                raise e

    def _train_epoch(self, device):
        self._before_train_epoch_start()
        if self.train_loader.sampler is not None and self.local_rank != -1:  # distributed
            if self.resume_index > 0:
                self.train_loader.sampler.set_epoch(epoch=self.epoch, resume_index=self.resume_index)
                print_log(f"Resume training from epoch {self.epoch}, global step {self.global_step}")
                self.resume_index = 0
            else:
                self.train_loader.sampler.set_epoch(self.epoch)
        t_iter = tqdm(enumerate(self.train_loader)) if self._is_main_proc() else enumerate(self.train_loader)
        metric_dict = defaultdict(list)
        print(f"NUMBATCHES = {len(self.train_loader)}")
        for batch_idx, batch in t_iter:
            try:
                batch = self.to_device(batch, device)
                loss = self.train_step(batch, self.global_step)
                if loss is None:
                    continue # Out of memory
                self.optimizer.zero_grad()
                loss.backward()
                metric_dict["loss"].append(loss.detach().cpu().item())
                if self.use_wandb and self._is_main_proc():
                    wandb.log({f'train_loss': loss.detach().cpu().item()}, step=self.global_step)
                    if batch_idx % 500 == 0 and batch_idx > 0:
                        start_idx = max(0, len(metric_dict["loss"]) - 500)
                        wandb.log({f'train_last500_loss': np.mean(metric_dict["loss"][start_idx:])}, step=self.global_step)
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()
                if hasattr(t_iter, 'set_postfix'):
                    t_iter.set_postfix(loss=loss.detach().cpu().item(), version=self.version)
                self.global_step += 1
                if self.sched_freq == 'batch':
                    self.scheduler.step()
                if self.use_wandb and self._is_main_proc():
                    wandb.log({f'lr': self.optimizer.param_groups[-1]['lr']}, step=self.global_step)
                if batch_idx == len(self.train_loader)//2:
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
            wandb.log({f'train_epoch_loss': np.mean(metric_dict["loss"])}, step=self.global_step)
        if self.sched_freq == 'epoch':
            self.scheduler.step()
    
    def _get_training_state(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "resume_index": ((self.global_step+1) % len(self.train_loader)) *self.train_loader.batch_size, 
            "global_step": self.global_step+1,
        }
    
    def _valid_epoch(self, device):
        if self.valid_loader is None:
            if self._is_main_proc():
                save_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.ckpt')
                if not os.path.exists(self.training_state_dir):
                    os.makedirs(self.training_state_dir)
                training_save_path = os.path.join(self.training_state_dir, f'training_state_epoch{self.epoch}_step{self.global_step}.pt')
                weights_path = os.path.join(self.model_dir, f'epoch{self.epoch}_step{self.global_step}.pt')
                config_path = os.path.join(self.model_dir, 'config.json')                    
                module_to_save = self.model.module if self.local_rank == 0 else self.model
                if self.config.save_topk < 0 or (self.config.max_epoch - self.epoch <= self.config.save_topk):
                    print_log(f'No validation, save path: {save_path}')
                    torch.save(module_to_save, save_path)
                    torch.save(self._get_training_state(), training_save_path)
                    torch.save(module_to_save.state_dict(), weights_path)
                    with open(config_path, 'w') as fout:
                        json.dump(module_to_save.get_config(), fout, indent=4)
                else:
                    print_log('No validation')
            return

        metric_dict = defaultdict(list)
        self.model.eval()
        with torch.no_grad():
            t_iter = tqdm(self.valid_loader) if self._is_main_proc() else self.valid_loader
            for batch in t_iter:
                batch = self.to_device(batch, device)
                loss, logits = self.valid_step(batch, self.valid_global_step)
                if loss is None:
                    continue # Out of memory
                metric_dict["loss"].append(loss.detach().cpu().item())
                metric_dict["pred"].extend(logits.detach().cpu().argmax(dim=1).tolist())
                metric_dict["label"].extend(batch['masked_labels'].detach().cpu().tolist())
                self.valid_global_step += 1
        self.model.train()
        # judge
        valid_metric = np.mean(metric_dict["loss"])
        if self.use_wandb and self._is_main_proc():
            wandb.log({'val_loss': valid_metric}, step=self.global_step)
            wandb.log({'val_acc': accuracy_score(metric_dict["label"], metric_dict["pred"])}, step=self.global_step)
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
            training_save_path = os.path.join(self.training_state_dir, f'training_state_epoch{self.epoch}_step{self.global_step}.pt')
            if not os.path.exists(self.training_state_dir):
                os.makedirs(self.training_state_dir)
            torch.save(self._get_training_state(), training_save_path)
            print_log(f'Training state save path: {training_save_path}')
            print_log(f'Validation: {valid_metric}, save path: {save_path}')
        if self._metric_better(valid_metric):
            self.patience = self.config.patience
        else:
            self.patience -= 1
        self.last_valid_metric = valid_metric
        for name in self.writer_buffer:
            value = np.mean(self.writer_buffer[name])
            self.log(name, value, self.epoch)
        self.writer_buffer = {}