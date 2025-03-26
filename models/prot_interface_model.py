import torch.nn as nn
import torch.nn.functional as F
from .prediction_model import PredictionModel, PredictionReturnValue
from .pretrain_model import DenoisePretrainModel
import torch
from copy import deepcopy
import json

class ProteinInterfaceModel(nn.Module):
    def __init__(self, model: PredictionModel) -> None:
        super().__init__()
        self.temp = 0.5
        self.cmplx_model = model
        self.prot_model = deepcopy(model)
        self.cmplx_model.requires_grad_(requires_grad=False)
        self.prot_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.prot_model.hidden_size, self.prot_model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.prot_model.hidden_size, self.prot_model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.prot_model.hidden_size, self.prot_model.hidden_size),
        )
        self.cmplx_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.cmplx_model.hidden_size, self.cmplx_model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cmplx_model.hidden_size, self.cmplx_model.hidden_size),
            nn.ReLU(),
            nn.Linear(self.cmplx_model.hidden_size, self.cmplx_model.hidden_size),
        )
    
    @classmethod
    def _load_from_pretrained(cls, pretrained_model, **kwargs):
        if pretrained_model.k_neighbors != kwargs.get('k_neighbors', pretrained_model.k_neighbors):
            print(f"Warning: pretrained model k_neighbors={pretrained_model.k_neighbors}, new model k_neighbors={kwargs.get('k_neighbors')}")
        model = PredictionModel(
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
        
        return cls(model)
    
    def get_config(self):
        return {
            'model_type': self.__class__.__name__,
            'prot_model': self.prot_model.get_config(),
        }
    
    @classmethod
    def load_from_pretrained(cls, pretrain_ckpt, **kwargs):
        pretrained_model = torch.load(pretrain_ckpt, map_location='cpu')
        if isinstance(pretrained_model, cls):
            return pretrained_model
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
        elif model_type == 'PredictionModel':
            pretrained_model = PredictionModel.load_from_config_and_weights(config_path, weights_path)
            return cls._load_from_pretrained(pretrained_model, **kwargs)
        elif model_type == cls.__name__:
            model_config = config['prot_model']
            assert model_config['model_type'] == 'PredictionModel', f"Model type {model_config['model_type']} not recognized for ProteinInterfaceModel"
            del model_config['model_type']
            model = PredictionModel(**model_config)
            pretrained_model = cls(model)
            pretrained_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            return pretrained_model
        else:
            raise ValueError(f"Model type {model_type} not recognized")
    
    def forward(self, batch_cmplx, batch_prot) -> PredictionReturnValue:
        with torch.no_grad():
            return_value_cmplx = self.cmplx_model.forward(
                batch_cmplx["X"], batch_cmplx["B"], batch_cmplx["A"], batch_cmplx["block_lengths"], batch_cmplx["lengths"], batch_cmplx["segment_ids"]
            )
        return_value_prot = self.prot_model.forward(
            batch_prot["X"], batch_prot["B"], batch_prot["A"], batch_prot["block_lengths"], batch_prot["lengths"], batch_prot["segment_ids"]
        )

        cmplx_repr = self.cmplx_ffn(return_value_cmplx.graph_repr)
        prot_repr = self.prot_ffn(return_value_prot.graph_repr)

        cmplx_repr = F.normalize(cmplx_repr, p=2, dim=-1) 
        prot_repr = F.normalize(prot_repr, p=2, dim=-1)

        loss = calculate_contrastive_loss(prot_repr, cmplx_repr, temperature=self.temp)
        return loss
    
    def infer(self, batch_prot, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch_prot['X'], B=batch_prot['B'], A=batch_prot['A'],
            block_lengths=batch_prot['block_lengths'],
            lengths=batch_prot['lengths'],
            segment_ids=batch_prot['segment_ids'],
        )
        prot_repr = self.prot_ffn(return_value.graph_repr)
        if extra_info:
            return prot_repr, return_value
        return prot_repr


def calculate_contrastive_loss(z, augmented_z, temperature=0.5, device="cpu"):
    # Contrastive loss
    z = F.normalize(z, dim=1).to(device)
    augmented_z = F.normalize(augmented_z, dim=1).to(device)
    diag_mask = torch.eye(z.size(0), dtype=torch.bool).to(device)

    logits_ab = torch.matmul(z, augmented_z.t()) / temperature
    logits_aa = torch.matmul(z, z.t()) / temperature
    similarity_pos_ab = torch.diag(logits_ab)
    similarity_negs_aa = logits_aa[~diag_mask].view(logits_aa.size(0), -1)
    contrastive_loss_ab = -torch.log(
        torch.exp(similarity_pos_ab)
        / (
            torch.sum(torch.exp(similarity_negs_aa), dim=1)
            + torch.sum(torch.exp(logits_ab), dim=1)
        )
    )
    logits_ba = torch.matmul(augmented_z, z.t()) / temperature
    logits_bb = torch.matmul(augmented_z, augmented_z.t()) / temperature
    similarity_pos_ba = torch.diag(logits_ba)
    similarity_negs_bb = logits_bb[~diag_mask].view(logits_bb.size(0), -1)
    contrastive_loss_ba = -torch.log(
        torch.exp(similarity_pos_ba)
        / (
            torch.sum(torch.exp(similarity_negs_bb), dim=1)
            + torch.sum(torch.exp(logits_ba), dim=1)
        )
    )
    return torch.mean(torch.cat((contrastive_loss_ab, contrastive_loss_ba)))