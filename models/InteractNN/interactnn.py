import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from .utils import TensorProductConvLayer, GaussianEmbedding


class InteractionModule(torch.nn.Module):
    def __init__(
        self,
        ns, # hidden dim of scalar features
        nv, # hidden dim of vector features
        num_conv_layers,
        sh_lmax,
        edge_size,
        dropout=0.0,
        norm_type="layer",
        return_noise=False,
    ):
        super(InteractionModule, self).__init__()
        self.ns, self.nv = ns, nv
        self.edge_size = edge_size
        self.num_conv_layers = num_conv_layers
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.edge_embedder = nn.Sequential(
            GaussianEmbedding(num_gaussians=edge_size),
            nn.Linear(edge_size, edge_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_size, edge_size),
        )
        self.node_embedding_dim = (
            ns if self.num_conv_layers < 3 else 2 * ns
        )  # only use the scalar and pseudo scalar features

        irrep_seq = [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o",
            f"{ns}x0e + {nv}x1o + {nv}x1e",
            f"{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o",
        ]

        conv_layers = []
        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                "in_irreps": in_irreps,
                "sh_irreps": self.sh_irreps,
                "out_irreps": out_irreps,
                "n_edge_features": 2 * ns + 2 * edge_size,  # features are [edge_length_embedding, edge_attr, scalars of atom 1, scalars of atom 2]
                "hidden_features": 2 * ns + 2 * edge_size,
                "residual": False,
                "norm_type": norm_type,
                "dropout": dropout,
            }
            conv_layers.append(TensorProductConvLayer(**parameters))

        self.norm_type = norm_type
        self.layers = nn.ModuleList(conv_layers)

        self.return_noise = return_noise
        if return_noise:
            denoise_parameters = {
                "in_irreps": irrep_seq[min(num_conv_layers, len(irrep_seq) - 1)],
                "sh_irreps": self.sh_irreps,
                "out_irreps": "1o + 1e",
                "n_edge_features": 2 * ns + 2 * edge_size,  # features are [edge_length_embedding, edge_attr, scalars of atom 1, scalars of atom 2]
                "hidden_features": 2 * ns + 2 * edge_size,
                "residual": False,
                "norm_type": norm_type,
                "dropout": dropout,
            }
            self.denoise_predictor = TensorProductConvLayer(
                **denoise_parameters
            )
            self.denoise_predictor.norm_layer.affine_bias.requires_grad = False # when predicting noise, there are no scalar irreps so this parameter is not needed
        else:
            self.out_ffn = nn.Sequential(
                nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.node_embedding_dim, ns),
            )


    def forward(self, node_attr, coords, batch_id, edges, edge_type_attr):
        edge_vec = coords[edges[1]] - coords[edges[0]]
        edge_sh = o3.spherical_harmonics(
            self.sh_irreps,
            edge_vec,
            normalize=True,
            normalization="component",
        )
        edge_length = edge_vec.norm(dim=-1)
        edge_length_embedding = self.edge_embedder(edge_length)

        for l in range(self.num_conv_layers):
            edge_attr = torch.cat(
                (
                    edge_length_embedding,
                    edge_type_attr,
                    node_attr[edges[0], : self.ns],
                    node_attr[edges[1], : self.ns],
                ),
                dim=1,
            )

            update = self.layers[l](
                node_attr, edges, edge_attr, edge_sh,
            )
            node_attr = F.pad(node_attr, (0, update.shape[-1]-node_attr.shape[-1])) 

            # update features with residual updates
            node_attr = node_attr + update

        if self.num_conv_layers < 3:
            node_embeddings = node_attr[:, : self.ns]
        else:
            node_embeddings = torch.cat(
                (
                    node_attr[:, : self.ns],
                    node_attr[:, -self.ns :],
                ),
                dim=1,
            )

        if self.return_noise:
            edge_attr = torch.cat(
                (
                    edge_length_embedding,
                    edge_type_attr,
                    node_attr[edges[0], : self.ns],
                    node_attr[edges[1], : self.ns],
                ),
                dim=1,
            )
            pred = self.denoise_predictor(
                node_attr, edges, edge_attr, edge_sh,
            )
            noise = pred[:, :3] + pred[:, 3:]
            return node_embeddings, noise
        else:
            node_embeddings = self.out_ffn(node_embeddings)
            return node_embeddings

