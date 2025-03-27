import torch.nn.functional as F
import torch
from torch_scatter import scatter_mean
import json

from .pretrain_model import DenoisePretrainModel
from data.pdb_utils import VOCAB
from .ATOMICA.utils import batchify


class MaskedNodeModel(DenoisePretrainModel):

    def __init__(self, atom_hidden_size, block_hidden_size, edge_size, k_neighbors,
                 n_layers, num_masked_block_classes, dropout=0.0, bottom_global_message_passing=False, 
                 global_message_passing=False, fragmentation_method=None) -> None:
        super().__init__(
            atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size, 
            k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, 
            bottom_global_message_passing=bottom_global_message_passing, global_message_passing=global_message_passing,
            atom_noise=False, translation_noise=False, rotation_noise=False, 
            torsion_noise=False, fragmentation_method=fragmentation_method, num_masked_block_classes=num_masked_block_classes)
        assert not any([self.atom_noise, self.translation_noise, self.rotation_noise, self.torsion_noise]), 'Masking model should not have any denoising heads'

    @classmethod
    def _load_from_pretrained(cls, pretrained_model, **kwargs):
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = cls(
            atom_hidden_size=pretrained_model.atom_hidden_size,
            block_hidden_size=pretrained_model.hidden_size,
            edge_size=pretrained_model.edge_size,
            k_neighbors=kwargs.get('k_neighbors', pretrained_model.k_neighbors),
            n_layers=pretrained_model.n_layers,
            dropout=kwargs.get('dropout', pretrained_model.dropout),
            fragmentation_method=pretrained_model.fragmentation_method if hasattr(pretrained_model, "fragmentation_method") else None, # for backward compatibility
            bottom_global_message_passing=kwargs.get('bottom_global_message_passing', pretrained_model.bottom_global_message_passing),
            global_message_passing=kwargs.get('global_message_passing', pretrained_model.global_message_passing),
            num_masked_block_classes=kwargs['num_masked_block_classes'],
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        if pretrained_model.global_message_passing is False and model.global_message_passing is True:
            model.edge_embedding_top.requires_grad_(requires_grad=True)
            print("Warning: global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        
        if pretrained_model.bottom_global_message_passing is False and model.bottom_global_message_passing is True:
            model.edge_embedding_bottom.requires_grad_(requires_grad=True)
            print("Warning: bottom_global_message_passing is True in the new model but False in the pretrain model, training edge_embedders in the model")
        return model
    
    def get_config(self):
        return {
            'atom_hidden_size': self.atom_hidden_size,
            'block_hidden_size': self.hidden_size,
            'edge_size': self.edge_size,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
            'k_neighbors': self.k_neighbors,
            'global_message_passing': self.global_message_passing,
            'bottom_global_message_passing': self.bottom_global_message_passing,
            'fragmentation_method': self.fragmentation_method,
            'num_masked_block_classes': self.num_masked_block_classes,
            'model_type': self.__class__.__name__,
        }

    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model: DenoisePretrainModel = torch.load(pretrain_ckpt, map_location='cpu')
        return cls._load_from_pretrained(pretrained_model, **kwargs)
    
    @classmethod
    def load_from_config_and_weights(cls, config_path, weights_path, **kwargs):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = config['model_type']
        del config['model_type']

        if model_type == 'DenoisePretrainModel':
            pretrained_model = DenoisePretrainModel.load_from_config_and_weights(config_path, weights_path)
            return cls._load_from_pretrained(pretrained_model, **kwargs)
        elif model_type == cls.__name__:
            pretrained_model = cls(**config)
            pretrained_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            return pretrained_model
        else:
            raise ValueError(f"Model type {model_type} not recognized")
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, masked_blocks, masked_labels, return_logits=False):
        with torch.no_grad():
            batch_id = torch.zeros_like(segment_ids)  # [Nb]
            batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
            batch_id.cumsum_(dim=0)  # [Nb], item idx in the batch

            block_id = torch.zeros_like(A) # [Nu]
            block_id[torch.cumsum(block_lengths, dim=0)[:-1]] = 1
            block_id.cumsum_(dim=0)  # [Nu], block (residue) id of each unit (atom)

            # transform blocks to single units
            bottom_batch_id = batch_id[block_id]  # [Nu]
            bottom_B = B[block_id]  # [Nu]
            bottom_segment_ids = segment_ids[block_id]  # [Nu]
            bottom_block_id = torch.arange(0, len(block_id), device=block_id.device)  #[Nu]


        # embedding
        bottom_H_0 = self.block_embedding.atom_embedding(A)
        top_H_0 = self.block_embedding.block_embedding(B)

        # bottom level message passing
        edges, edge_attr = self.get_edges(bottom_B, bottom_batch_id, bottom_segment_ids, 
                                          Z, bottom_block_id, self.bottom_global_message_passing, 
                                          top=False)
        bottom_block_repr = self.encoder(bottom_H_0, Z, bottom_batch_id, None, edges, edge_attr)
        
        # top level message passing 
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)

        if self.bottom_global_message_passing:
            batched_bottom_block_repr, _ = batchify(bottom_block_repr, block_id)
        else:
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, _ = batchify(bottom_block_repr[atom_mask], block_id[atom_mask])

        block_repr_from_bottom = self.atom_block_attn(top_H_0.unsqueeze(1), batched_bottom_block_repr)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = self.atom_block_attn_norm(top_H_0)

        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr) 
        
        logits = self.masked_ffn(block_repr[masked_blocks])
        masked_loss = F.cross_entropy(logits, masked_labels)
        if return_logits:
            return masked_loss, logits
        pred_blocks = F.softmax(logits, dim=1)
        return masked_loss, pred_blocks
    
    def infer(self, batch):
        self.eval()
        loss, pred_blocks = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            masked_blocks=batch['masked_blocks'],
            masked_labels=batch['masked_labels'],
        )
        return pred_blocks