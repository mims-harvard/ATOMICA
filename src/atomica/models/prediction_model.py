from collections import namedtuple
import torch
from ..utils.scatter import scatter_mean

from ..data.pdb_utils import VOCAB
from .pretrain_model import DenoisePretrainModel
from .atomica.utils import batchify
import json

_PredictionReturnValueBase = namedtuple(
    'ReturnValue',
    ['unit_repr', 'block_repr', 'graph_repr', 'batch_id', 'block_id',
     'block_node_attr', 'atom_node_attr',
     'atom_invariant_repr', 'block_invariant_repr', 'graph_invariant_repr'],
    # fields after `block_id` default to None, so callers of the first five are unaffected
    defaults=[None, None, None, None, None],
)


class PredictionReturnValue(_PredictionReturnValueBase):
    """Model outputs in the paper's two families, both SE(3)-invariant.

    ``unit_repr`` / ``block_repr`` / ``graph_repr`` are h^atom, h^block and h^graph, built from
    the l=0 channels. ``*_invariant_repr`` are z^atom, z^block and z^graph, which add invariants
    from the l>0 channels; they are filled only with ``return_invariant_repr=True``.

    :mod:`atomica.representations` is the supported way to read these.
    """

    @property
    def atom_scalar_repr(self):
        """Alias of ``unit_repr`` (the paper's h^atom)."""
        return self.unit_repr

    @property
    def block_scalar_repr(self):
        """Alias of ``block_repr`` (the paper's h^block)."""
        return self.block_repr

    @property
    def graph_scalar_repr(self):
        """Alias of ``graph_repr`` (the paper's h^graph)."""
        return self.graph_repr

class PredictionModel(DenoisePretrainModel):
    def __init__(self, atom_hidden_size, block_hidden_size, edge_size, k_neighbors,
                 n_layers, dropout=0.0, bottom_global_message_passing=False, global_message_passing=False, fragmentation_method=None) -> None:
        super().__init__(
            atom_hidden_size=atom_hidden_size, block_hidden_size=block_hidden_size, edge_size=edge_size, 
            k_neighbors=k_neighbors, n_layers=n_layers, dropout=dropout, 
            bottom_global_message_passing=bottom_global_message_passing, global_message_passing=global_message_passing,
            atom_noise=False, translation_noise=False, rotation_noise=False, 
            torsion_noise=False, fragmentation_method=fragmentation_method, num_masked_block_classes=None)
        assert not any([self.atom_noise, self.translation_noise, self.rotation_noise, self.torsion_noise]), 'Prediction model should not have any denoising heads'
    
    @classmethod
    def _load_from_pretrained(cls, pretrained_model: DenoisePretrainModel, **kwargs):
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
        )
        print(f"""Pretrained model params: hidden_size={model.hidden_size},
               edge_size={model.edge_size}, k_neighbors={model.k_neighbors}, 
               n_layers={model.n_layers}, bottom_global_message_passing={model.bottom_global_message_passing},
               global_message_passing={model.global_message_passing}, 
               fragmentation_method={model.fragmentation_method}""")
        assert not any([model.atom_noise, model.translation_noise, model.rotation_noise, model.torsion_noise]), "prediction model no noise"
        model.load_state_dict(pretrained_model.state_dict(), strict=False)

        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.requires_grad_(requires_grad=False)

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
            'model_type': self.__class__.__name__,
        }

    @classmethod
    def load_from_config_and_weights(cls, config_path, weights_path, **kwargs):
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_type = config['model_type']
        del config['model_type']

        if 'nonlinearity' in config:
            if config['nonlinearity'] == 'relu':
                config["nonlinearity"] = torch.nn.ReLU()
            elif config['nonlinearity'] == 'gelu':
                config["nonlinearity"] = torch.nn.GELU()
            elif config['nonlinearity'] == 'elu':
                config["nonlinearity"] = torch.nn.ELU()
            else:
                raise NotImplementedError(f"Nonlinearity {config['nonlinearity']} not implemented")

        # Add default focal loss parameters for backward compatibility with old models
        # If these are not in the config (old models), use defaults
        if model_type in ['MultiClassClassifierModel', 'MultiLabelClassifierModel']:
            if 'loss_type' not in config:
                config['loss_type'] = 'binary_cross_entropy' if model_type == 'MultiLabelClassifierModel' else 'cross_entropy'
            if 'focal_alpha' not in config:
                config['focal_alpha'] = None
            if 'focal_gamma' not in config:
                config['focal_gamma'] = 2.0

        if model_type == 'DenoisePretrainModel':
            pretrained_model = DenoisePretrainModel.load_from_config_and_weights(config_path, weights_path)
            return cls._load_from_pretrained(pretrained_model, **kwargs)
        elif model_type == cls.__name__:
            pretrained_model = cls(**config)
            pretrained_model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
            return pretrained_model
        else:
            raise ValueError(f"Model type {model_type} not recognized")

    ########## overload ##########
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, return_graph_repr=True,
                return_full_node_attr=False, return_invariant_repr=False,
                invariant_pool='mean_std_global') -> PredictionReturnValue:
        """Run the encoder stack.

        ``return_invariant_repr=True`` also fills the ``z`` family fields.
        """
        if return_invariant_repr:
            # the invariant descriptors are built from the raw irrep tensors
            return_full_node_attr = True
        # batch_id and block_id
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
        if return_full_node_attr:
            # atom-level irrep tensor, needed to build z^atom
            bottom_block_repr, atom_node_attr = self.encoder(
                bottom_H_0, Z, bottom_batch_id, None, edges, edge_attr,
                return_full_node_attr=True,
            )
        else:
            atom_node_attr = None
            bottom_block_repr = self.encoder(
                bottom_H_0, Z, bottom_batch_id, None, edges, edge_attr,
            )
        
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
        if return_full_node_attr:
            # block-level irrep tensor, needed to build z^block
            block_repr, block_node_attr = self.top_encoder(
                top_H_0, top_Z, batch_id, None, edges, edge_attr, return_full_node_attr=True)
        else:
            block_node_attr = None
            block_repr = self.top_encoder(top_H_0, top_Z, batch_id, None, edges, edge_attr)
        if return_graph_repr:
            if self.global_message_passing:
                graph_repr = self.attention_pooling(block_repr, batch_id)
            else:
                global_mask = B != self.global_block_id
                graph_repr = self.attention_pooling(block_repr[global_mask], batch_id[global_mask])
        else:
            graph_repr = None


        # ---- z family (None unless return_invariant_repr=True) ----------------------------
        atom_invariant_repr = block_invariant_repr = graph_invariant_repr = None
        if return_invariant_repr:
            from .atomica.invariants import pool_atoms_to_blocks
            from ..representations import pool_blocks
            inv = self._irrep_invariants()
            inv.check_dim(block_node_attr)
            # z^atom: 0e/0o channels (+) within-degree Gram of the l>0 channels
            atom_invariant_repr = inv.descriptor(atom_node_attr)
            # z^block: h^block (+) block Gram (+) mean and s.d. of z^atom over the block's atoms.
            # block_id is a global atom->block index, so this is correct for a batched forward.
            block_invariant_repr = torch.cat([
                block_repr,
                inv.gram(block_node_attr),
                pool_atoms_to_blocks(atom_invariant_repr, block_id,
                                     n_blocks=block_node_attr.shape[0]),
            ], dim=-1)
            # z^graph: parameter-free pooling, so a trained head stays the only fitted component.
            # invariant_pool=None means the caller wants only the atom- or block-level invariants,
            # so the graph vector is skipped rather than pooled by an arbitrary default rule.
            if invariant_pool is not None:
                graph_invariant_repr = pool_blocks(
                    block_invariant_repr, batch_id, B == self.global_block_id,
                    mode=invariant_pool, component_dims=self.invariant_component_dims())

        return PredictionReturnValue(
            # h family
            unit_repr=bottom_block_repr,
            block_repr=block_repr,
            graph_repr=graph_repr,

            # batch information
            batch_id=batch_id,
            block_id=block_id,

            # raw irrep tensors (None unless return_full_node_attr=True)
            block_node_attr=block_node_attr,
            atom_node_attr=atom_node_attr,

            # z family (None unless return_invariant_repr=True)
            atom_invariant_repr=atom_invariant_repr,        # [Nu, 2*ns + n_gram]
            block_invariant_repr=block_invariant_repr,      # [Nb, ns + n_gram + 2*(2*ns + n_gram)]
            graph_invariant_repr=graph_invariant_repr,      # [n_graphs, k * block width]
        )

    def _irrep_invariants(self):
        """Cached irrep layout for this model. Derived from the encoder, never hand-specified."""
        if getattr(self, '_invariants_cache', None) is None:
            from .atomica.invariants import IrrepInvariants
            self._invariants_cache = IrrepInvariants.from_encoder(self.top_encoder)
        return self._invariants_cache

    def invariant_component_dims(self):
        """Widths of the three parts of z^block, in order: h^block, block Gram, atom-pooled.

        Downstream code needs these to slice z^block back apart, and the
        ``mean_component_normalized`` pooling rule needs them to normalize each part separately.
        """
        inv = self._irrep_invariants()
        ns = self.top_encoder.encoder.ns
        return {"h_block": int(ns), "gram": int(inv.n_gram), "atom": int(2 * inv.n_descriptor)}

    def infer(self, batch, return_invariant_repr=False, invariant_pool='mean_std_global'):
        self.eval()
        return_value = self.forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
            return_invariant_repr=return_invariant_repr,
            invariant_pool=invariant_pool,
        )
        return return_value