import torch
from torch import nn
from e3nn import o3
from torch_scatter import scatter
import torch.nn.functional as F
from e3nn.nn import BatchNorm
import torch
import torch.nn as nn
from e3nn.o3 import Irreps


# Source: https://github.com/atomicarchitects/equiformer/blob/master/nets/layer_norm.py
# Using EquivariantLayerNormV2
class EquivariantLayerNorm(nn.Module):
    def __init__(self, irreps, eps=1e-5, affine=True, normalization="component"):
        super().__init__()

        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__}({self.irreps}, eps={self.eps})"

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, node_input, **kwargs):
        # batch, *size, dim = node_input.shape  # TODO: deal with batch
        # node_input = node_input.reshape(batch, -1, dim)  # [batch, sample, stacked features]
        # node_input has shape [batch * nodes, dim], but with variable nr of nodes.
        # the node_input batch slices this into separate graphs
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            # field = node_input[:, ix: ix + mul * d]  # [batch * sample, mul * repr]
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            # [batch * sample, mul, repr]
            field = field.reshape(-1, mul, d)

            # For scalars first compute and subtract the mean
            if ir.l == 0 and ir.p == 1:
                # Compute the mean
                field_mean = torch.mean(field, dim=1, keepdim=True)  # [batch, mul, 1]]
                # Subtract the mean
                field = field - field_mean

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)

            # Then apply the rescaling (divide by the sqrt of the squared_norm, i.e., divide by the norm
            field_norm = (field_norm + self.eps).pow(-0.5)  # [batch, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]  # [batch, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch, mul]

            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            if self.affine and d == 1 and ir.p == 1:  # scalars
                bias = self.affine_bias[ib : ib + mul]  # [batch, mul]
                ib += mul
                field += bias.reshape(mul, 1)  # [batch * sample, mul, repr]

            # Save the result, to be stacked later with the rest
            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        if ix != dim:
            fmt = (
                "`ix` should have reached node_input.size(-1) ({}), but it ended at {}"
            )
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        output = torch.cat(fields, dim=-1)  # [batch * sample, stacked features]
        return output


class TensorProductConvLayer(torch.nn.Module):
    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=True,
        norm_type="layer",
        dropout=0.0,
        hidden_features=None,
    ):
        super(TensorProductConvLayer, self).__init__()
        assert norm_type in [
            "layer",
            "batch",
            "none",
        ], "supported norm_types are layer or batch or none"
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(
            in_irreps, sh_irreps, out_irreps, shared_weights=False
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel),
        )
        self.norm_type = norm_type
        if self.norm_type == "layer":
            self.norm_layer = EquivariantLayerNorm(out_irreps)
        elif self.norm_type == "batch":
            self.norm_layer = BatchNorm(out_irreps)
        else:
            self.norm_type = "none"
            self.norm_layer = None

    def forward(
        self,
        node_attr,
        edge_index,
        edge_attr,
        edge_sh,
        node_attr_dst=None,
        out_nodes=None,
        reduce="mean",
    ):
        edge_src, edge_dst = edge_index
        edge_feat = self.fc(edge_attr)
        assert not torch.any(torch.isnan(edge_feat)), "nans in edge_feat"
        assert not torch.any(torch.isnan(edge_sh)), "nans in edge_sh"
        assert not torch.any(torch.isnan(node_attr)), "nans in node_attr"
        tp = self.tp(
            node_attr[edge_dst] if node_attr_dst is None else node_attr_dst,
            edge_sh,
            edge_feat,
        )  # weighted tensor product of edge features and edge sh
        assert not torch.any(torch.isnan(tp)), "nans in tp"
        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(
            tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce
        )  # mean over all neighbours
        assert not torch.any(torch.isnan(out)), "nans in out"

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = (
                out + padded
            )  # h_a (padded) + message from neighbours (out), if no residual then it is just the message from neighbours

        if self.norm_layer is not None:  # FIXME: commented for debugging
            out = self.norm_layer(out)
        return out


class GaussianEmbedding(torch.nn.Module):
    # used to embed the edge distances
    # NOTE stop should be the max edge length in the dataset
    def __init__(self, start=0.0, stop=5.0, num_gaussians=32):
        super().__init__()
        self.embedding_dim = num_gaussians
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        original_shape = dist.shape
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        dist = dist.reshape(*original_shape, self.embedding_dim)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class AtomEncoder(nn.Module):
    # From SchNet
    def __init__(self, dim, trainable=True):
        super(AtomEncoder, self).__init__()
        self._dim = dim
        self._trainable = trainable
        self._elements = range(119)  # 0 = masked element, 1-118 = elements
        self.embeddings = nn.Embedding(len(self._elements), self._dim)

        if not self._trainable:
            self.embeddings.weight.requires_grad = False

    def forward(self, elems):
        # FIXME: now that the elements are just range(1, 119) we can just directly use elems
        # leaving for now for backwards compatibility
        y = self.embeddings(elems)
        return y


class SphericalHarmonicEdgeAttrs(nn.Module):
    def __init__(
        self,
        irreps_edge_sh: int,
        edge_sh_normalization: str = "component",
        edge_sh_normalize: bool = True,
    ):
        super(SphericalHarmonicEdgeAttrs, self).__init__()
        self.irreps_edge_sh = o3.Irreps.spherical_harmonics(irreps_edge_sh)
        self.sh = o3.SphericalHarmonics(
            self.irreps_edge_sh, edge_sh_normalize, edge_sh_normalization
        )
    
    def forward(self, edge_vec):
        return self.sh(edge_vec)


def batchify(tensor, batch_ids, max_seq_len=None):
    # Determine the number of batches and the maximum sequence length
    num_batches = batch_ids.max().item() + 1
    max_seq_len = (batch_ids == torch.arange(num_batches, device=batch_ids.device).unsqueeze(1)).sum(dim=1).max().item() if max_seq_len is None else max_seq_len
    
    # Initialize the output tensor with the mask token
    _, dim = tensor.shape
    output = torch.zeros((num_batches, max_seq_len, dim), device=tensor.device) # * mask_token
    batchify_mask = torch.zeros((num_batches, max_seq_len), device=tensor.device, dtype=torch.bool)
    
    # Populate the output tensor and the mask
    for batch_id in range(num_batches):
        mask = batch_ids == batch_id
        sequence = tensor[mask]
        output[batch_id, :sequence.size(0)] = sequence
        batchify_mask[batch_id, :sequence.size(0)] = True
    
    # output is of shape (num_batches, max_seq_len, dim)
    return output, batchify_mask

def unbatchify(batchified_tensor, batchify_mask):
    # Get the dimensions
    num_batches, max_seq_len, dim = batchified_tensor.shape
    original_length = batchify_mask.sum().item()
    
    # Initialize the output tensor
    output = torch.zeros((original_length, dim), device=batchified_tensor.device)
    
    # Fill the output tensor using the mask
    idx = 0
    for batch_id in range(num_batches):
        seq_length = batchify_mask[batch_id].sum().item()
        output[idx:idx + seq_length] = batchified_tensor[batch_id, :seq_length]
        idx += seq_length
    
    return output