import torch.nn as nn
import torch.nn.functional as F
from .prediction_model import PredictionModel, PredictionReturnValue

class ClassifierModel(PredictionModel):

    def __init__(self, num_pred_layers, nonlinearity, pred_dropout, pred_hidden_size, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_pred_layers = num_pred_layers
        self.pred_dropout = pred_dropout
        self.pred_hidden_size = pred_hidden_size
        self.nonlinearity = 'relu' if isinstance(nonlinearity, nn.ReLU) else 'gelu' if nonlinearity == nn.GELU else 'elu' if nonlinearity == nn.ELU else None
        layers = [nonlinearity, nn.Dropout(pred_dropout), nn.Linear(self.hidden_size, pred_hidden_size)]
        for _ in range(0, num_pred_layers-2):
            layers.extend([nonlinearity, nn.Dropout(pred_dropout), nn.Linear(pred_hidden_size, pred_hidden_size)])
        layers.extend([nonlinearity, nn.Dropout(pred_dropout), nn.Linear(pred_hidden_size, 1)])
        self.classifier_ffn = nn.Sequential(*layers)
    
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
            nonlinearity=kwargs['nonlinearity'],
            num_pred_layers=kwargs['num_pred_layers'],
            pred_dropout=kwargs['pred_dropout'],
            pred_hidden_size=kwargs['pred_hidden_size'],
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
        model.attention_pooling.requires_grad_(requires_grad=False) # pooling is not used in affinity prediction
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.classifer_ffn.requires_grad_(requires_grad=True)
        return model

    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'num_pred_layers': self.num_pred_layers,
            'nonlinearity': self.nonlinearity,
            'pred_dropout': self.pred_dropout,
            'pred_hidden_size': self.pred_hidden_size,
        })
        return config_dict
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label, block_embeddings=None, block_embeddings0=None, block_embeddings1=None) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        logits = self.classifier_ffn(return_value.graph_repr).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        return loss, F.sigmoid(logits)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        logits = self.classifier_ffn(return_value.graph_repr)
        pred = F.sigmoid(logits)
        if extra_info:
            return pred, return_value
        return pred


class MultiClassClassifierModel(PredictionModel):

    def __init__(self, num_classes, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier_ffn = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.num_classes),
        )
    
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
            num_classes=kwargs['num_classes'],
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
        config_dict = super().get_config()
        config_dict['num_classes'] = self.num_classes
        return config_dict
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label, block_embeddings=None, block_embeddings0=None, block_embeddings1=None) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        logits = self.classifier_ffn(return_value.graph_repr)
        prob = F.softmax(logits, dim=1)
        return F.cross_entropy(logits, label), prob
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        logits = self.classifier_ffn(return_value.graph_repr)
        pred_label = F.softmax(logits, dim=1)
        if extra_info:
            return pred_label, return_value
        return pred_label


class RegressionPredictor(PredictionModel):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        nonlinearity = nn.ReLU
        self.energy_ffn = nn.Sequential(
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nonlinearity(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    @classmethod
    def _load_from_pretrained(cls, pretrained_model, **kwargs):
        model = super()._load_from_pretrained(pretrained_model, **kwargs)
        partial_finetune = kwargs.get('partial_finetune', False)
        if partial_finetune:
            model.energy_ffn.requires_grad_(requires_grad=True)
        return model
    
    def forward(self, Z, B, A, block_lengths, lengths, segment_ids, label, block_embeddings=None, block_embeddings0=None, block_embeddings1=None) -> PredictionReturnValue:
        return_value = super().forward(Z, B, A, block_lengths, lengths, segment_ids)
        pred_energy = self.energy_ffn(return_value.graph_repr).squeeze(-1)
        return F.mse_loss(pred_energy, label), pred_energy  # since we are supervising pK=-log_10(Kd), whereas the energy is RTln(Kd)
    
    def infer(self, batch, extra_info=False):
        self.eval()
        return_value = super().forward(
            Z=batch['X'], B=batch['B'], A=batch['A'],
            block_lengths=batch['block_lengths'],
            lengths=batch['lengths'],
            segment_ids=batch['segment_ids'],
        )
        pred_energy = self.energy_ffn(return_value.graph_repr).squeeze(-1)
        if extra_info:
            return pred_energy, return_value
        return pred_energy
    