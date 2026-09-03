import torch.nn.functional as F
import torch
import torch.nn as nn
from ..utils.scatter import scatter_mean, scatter_min, scatter_sum
import json

from .pretrain_model import DenoisePretrainModel
from ..data.pdb_utils import VOCAB
from .atomica.utils import batchify, GaussianEmbedding
from .tools import _unit_edges_from_block_edges


class MaskedNodeModel(DenoisePretrainModel):

    def __init__(self, atom_hidden_size, block_hidden_size, edge_size, k_neighbors,
                 n_layers, num_masked_block_classes, dropout=0.0, bottom_global_message_passing=False, 
                 global_message_passing=False, fragmentation_method=None,
                 top_max_edge_length=5, top_long_range_edge_length=None,
                 na_loss_weight=1.0, attn_pad_mask=False,
                 masked_affine=False, bottom_repr_scale=False,
                 top_pair_geom=False) -> None:
        """Masked block-type prediction head on top of the ATOMICA encoder.

        Every keyword after ``fragmentation_method`` defaults to the released checkpoint's
        behaviour, and the last four are the identity at initialisation:

        na_loss_weight     scale the training loss on nucleotide positions (labels 20-27)
        attn_pad_mask      keep ``batchify`` padding out of ``atom_block_attn``'s softmax
        masked_affine      per-channel affine on the masked block representation
        bottom_repr_scale  learnable scalar on the atoms entering ``atom_block_attn``
        top_pair_geom      per-atom-pair contact channel on block-level edges
        """
        super().__init__(
            atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size, 
            k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, 
            bottom_global_message_passing=bottom_global_message_passing, global_message_passing=global_message_passing,
            atom_noise=False, translation_noise=False, rotation_noise=False, 
            torsion_noise=False, fragmentation_method=fragmentation_method, num_masked_block_classes=num_masked_block_classes,
            top_max_edge_length=top_max_edge_length, top_long_range_edge_length=top_long_range_edge_length)
        assert not any([self.atom_noise, self.translation_noise, self.rotation_noise, self.torsion_noise]), 'Masking model should not have any denoising heads'
        self.na_loss_weight = na_loss_weight
        self.attn_pad_mask = attn_pad_mask

        self.masked_affine = masked_affine
        if masked_affine:
            self.masked_affine_weight = nn.Parameter(torch.ones(block_hidden_size))
            self.masked_affine_bias = nn.Parameter(torch.zeros(block_hidden_size))

        self.bottom_repr_scale = bottom_repr_scale
        if bottom_repr_scale:
            self.bottom_repr_scale_param = nn.Parameter(torch.ones(1))

        self.top_pair_geom = top_pair_geom
        if top_pair_geom:
            self.pair_geom_rbf = GaussianEmbedding(start=0.0, stop=12.0, num_gaussians=16)
            self.pair_geom_ffn = nn.Sequential(
                nn.Linear(16 * 2 + 3, edge_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_size, edge_size),
            )
            # zero the final Linear rather than the gate, so the branch still receives gradient
            nn.init.zeros_(self.pair_geom_ffn[-1].weight)
            nn.init.zeros_(self.pair_geom_ffn[-1].bias)
            self.pair_geom_gate = nn.Parameter(torch.ones(1))

    def _pair_geom_features(self, Z, block_id, edges):
        """Per-atom-pair contact summary for each block edge. Returns [E, 35].

        RBF of the minimum and mean atom-atom distance, plus soft contact counts in the
        hydrogen-bond, stacking and covalent windows.
        """
        E = edges.shape[1]
        (u_src, u_dst), (edge_id, _, _) = _unit_edges_from_block_edges(
            block_id, edges.t().contiguous())
        d = (Z[u_src] - Z[u_dst]).norm(dim=-1)  # [Eu]
        d_min = scatter_min(d, edge_id, dim_size=E)[0]
        d_mean = scatter_mean(d, edge_id, dim_size=E)
        tau = 0.15  # window softness, Angstrom

        def soft_count(lo, hi):
            w = torch.sigmoid((d - lo) / tau) * torch.sigmoid((hi - d) / tau)
            return torch.log1p(scatter_sum(w, edge_id, dim_size=E))

        counts = torch.stack([
            soft_count(2.6, 3.2),   # Watson-Crick / hydrogen bond
            soft_count(3.2, 4.0),   # base stacking
            soft_count(0.0, 2.0),   # covalent / sequential
        ], dim=-1)
        return torch.cat([self.pair_geom_rbf(d_min), self.pair_geom_rbf(d_mean), counts], dim=-1)

    @classmethod
    def _load_from_pretrained(cls, pretrained_model, **kwargs):
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        if kwargs.get('num_masked_block_classes',
                      getattr(pretrained_model, 'num_masked_block_classes', None)) is None:
            raise ValueError(
                f"{cls.__name__} needs num_masked_block_classes, and "
                f"{type(pretrained_model).__name__} carries none (it was pretrained without a "
                f"masked-block head). Pass it explicitly to fine-tune a fresh head.")
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
            # fall back to the pretrained model's own head width; the released checkpoint
            # carries a trained masked_ffn, so it can be evaluated without fine-tuning
            num_masked_block_classes=kwargs.get(
                'num_masked_block_classes',
                getattr(pretrained_model, 'num_masked_block_classes', None)),
            top_max_edge_length=kwargs.get(
                'top_max_edge_length', getattr(pretrained_model, 'top_max_edge_length', 5)),
            top_long_range_edge_length=kwargs.get('top_long_range_edge_length', None),
            na_loss_weight=kwargs.get('na_loss_weight', 1.0),
            attn_pad_mask=kwargs.get('attn_pad_mask', False),
            masked_affine=kwargs.get('masked_affine', False),
            bottom_repr_scale=kwargs.get('bottom_repr_scale', False),
            top_pair_geom=kwargs.get('top_pair_geom', False),
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
            'top_max_edge_length': self.top_max_edge_length,
            'top_long_range_edge_length': self.top_long_range_edge_length,
            'na_loss_weight': self.na_loss_weight,
            'attn_pad_mask': self.attn_pad_mask,
            'masked_affine': self.masked_affine,
            'bottom_repr_scale': self.bottom_repr_scale,
            'top_pair_geom': self.top_pair_geom,
            'num_masked_block_classes': self.num_masked_block_classes,
            'model_type': self.__class__.__name__,
        }

    @classmethod
    def _drop_inactive_config_keys(cls, config):
        """Drop config keys this class does not accept, but only where they are switched off.

        A key set to False, 0 or None describes a component that was never built, so dropping it
        loads the identical model. A key that is switched on raises instead.
        """
        import inspect
        accepted = set(inspect.signature(cls.__init__).parameters) - {'self'}
        out, unknown_on = {}, []
        for k, v in config.items():
            if k in accepted:
                out[k] = v
            elif v in (False, 0, 0.0, None):
                continue
            else:
                unknown_on.append(f"{k}={v!r}")
        if unknown_on:
            raise ValueError(
                f"{cls.__name__} cannot be built from this config: it enables options this "
                f"class does not implement ({', '.join(unknown_on)})")
        return out

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
            config = cls._drop_inactive_config_keys(config)
            pretrained_model = cls(**config)
            pretrained_model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
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
        if self.bottom_repr_scale:
            # inverse temperature on atom_block_attn's softmax; 1.0 at initialisation
            bottom_block_repr = self.bottom_repr_scale_param * bottom_block_repr
        
        # top level message passing 
        top_Z = scatter_mean(Z, block_id, dim=0)  # [Nb, n_channel, 3]
        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        edges, edge_attr = self.get_edges(B, batch_id, segment_ids, top_Z, top_block_id, 
                                          self.global_message_passing, top=True)
        if self.top_pair_geom:
            # the final projection is zero-initialised, so this term is exactly 0 at init
            edge_attr = edge_attr + self.pair_geom_gate * self.pair_geom_ffn(
                self._pair_geom_features(Z, block_id, edges))

        if self.bottom_global_message_passing:
            batched_bottom_block_repr, bt_mask = batchify(bottom_block_repr, block_id)
        else:
            atom_mask = A != VOCAB.get_atom_global_idx()
            batched_bottom_block_repr, bt_mask = batchify(bottom_block_repr[atom_mask], block_id[atom_mask])

        block_repr_from_bottom = self.atom_block_attn(
            top_H_0.unsqueeze(1), batched_bottom_block_repr,
            kv_mask=bt_mask if self.attn_pad_mask else None)
        top_H_0 = top_H_0 + block_repr_from_bottom.squeeze(1)
        top_H_0 = self.atom_block_attn_norm(top_H_0)

        top_block_id = torch.arange(0, len(batch_id), device=batch_id.device)
        block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr) 
        
        masked_repr = block_repr[masked_blocks]
        if self.masked_affine:
            # weight=1, bias=0 at initialisation, so this is the identity for a fresh model
            masked_repr = masked_repr * self.masked_affine_weight + self.masked_affine_bias
        logits = self.masked_ffn(masked_repr)
        if self.na_loss_weight == 1.0:
            masked_loss = F.cross_entropy(logits, masked_labels)
        else:
            # upweight nucleotide positions (labels 20-27); masking itself is unchanged
            per = F.cross_entropy(logits, masked_labels, reduction='none')
            w = torch.where((masked_labels >= 20) & (masked_labels <= 27),
                            torch.as_tensor(self.na_loss_weight, device=per.device, dtype=per.dtype),
                            torch.ones_like(per))
            masked_loss = (per * w).sum() / w.sum()
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