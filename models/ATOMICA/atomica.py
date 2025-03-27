import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from .utils import TensorProductConvLayer, GaussianEmbedding
from torch_scatter import scatter_mean
from torch_cluster import radius

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
        return_atom_noise=False,
        return_torsion_noise=False,
        return_global_noise=False,
        max_torsion_neighbors=9,
        max_edge_length=20, 
        max_global_edge_length=20,
        max_torsion_edge_length=5,
    ):
        super(InteractionModule, self).__init__()
        self.ns, self.nv = ns, nv
        self.edge_size = edge_size
        self.num_conv_layers = num_conv_layers
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.edge_embedder = nn.Sequential(
            GaussianEmbedding(num_gaussians=edge_size, stop=max_edge_length),
            nn.Linear(edge_size, edge_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_size, edge_size),
        )
        self.node_embedding_dim = (
            ns if self.num_conv_layers < 3 else 2 * ns
        )  # only use the scalar and pseudo scalar features

        irrep_seq = [
            f'{ns}x0e',
            f'{ns}x0e + {nv}x1o + {nv}x2e',
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
            f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o',
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

        self.return_atom_noise = return_atom_noise
        self.return_torsion_noise = return_torsion_noise
        self.return_global_noise = return_global_noise
        if return_global_noise:
            self.global_denoise_edge_embedder = nn.Sequential(
                GaussianEmbedding(num_gaussians=edge_size, stop=max_global_edge_length),
                nn.Linear(edge_size, edge_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_size, edge_size),
            )
            global_denoise_parameters = {
                "in_irreps": irrep_seq[min(num_conv_layers, len(irrep_seq) - 1)],
                "sh_irreps": self.sh_irreps,
                "out_irreps": "2x1o + 2x1e",
                "n_edge_features": ns + edge_size,  # features are [edge_length_embedding, edge_attr, scalars of atom 1, scalars of atom 2]
                "hidden_features": ns + edge_size,
                "residual": False,
                "norm_type": norm_type,
                "dropout": dropout,
            }
            self.global_denoise_predictor = TensorProductConvLayer(
                **global_denoise_parameters
            )
            self.global_denoise_predictor.norm_layer.affine_bias.requires_grad = False # when predicting noise, there are no scalar irreps so this parameter is not needed
        if return_torsion_noise:
            self.tor_max_radius = max_torsion_edge_length
            self.tor_max_neighbors = max_torsion_neighbors
            self.torsion_edge_embedder = nn.Sequential(
                GaussianEmbedding(num_gaussians=edge_size, stop=max_torsion_edge_length),
                nn.Linear(edge_size, edge_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_size, edge_size),
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=irrep_seq[min(num_conv_layers, len(irrep_seq) - 1)],
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f'{ns}x0o + {ns}x0e',
                n_edge_features=2 * ns + edge_size,
                residual=False,
                dropout=dropout,
                norm_type=norm_type,
            )
            self.tor_final_layer = nn.Sequential(
                nn.Linear(2 * ns, ns, bias=False),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False)
            )
        if return_atom_noise:
            self.local_denoise_edge_embedder = nn.Sequential(
                GaussianEmbedding(num_gaussians=edge_size, stop=max_edge_length),
                nn.Linear(edge_size, edge_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(edge_size, edge_size),
            )
            local_denoise_parameters = {
                "in_irreps": irrep_seq[min(num_conv_layers, len(irrep_seq) - 1)],
                "sh_irreps": self.sh_irreps,
                "out_irreps": "1o + 1e",
                "n_edge_features": 2 * ns + 2 * edge_size,  # features are [edge_length_embedding, edge_attr, scalars of atom 1, scalars of atom 2]
                "hidden_features": 2 * ns + 2 * edge_size,
                "residual": False,
                "norm_type": norm_type,
                "dropout": dropout,
            }
            self.local_denoise_predictor = TensorProductConvLayer(
                **local_denoise_parameters
            )
            self.local_denoise_predictor.norm_layer.affine_bias.requires_grad = False # when predicting noise, there are no scalar irreps so this parameter is not needed
        
        self.out_ffn = nn.Sequential(
            nn.Linear(self.node_embedding_dim, self.node_embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.node_embedding_dim, ns),
        )
    
    def remove_torsion_denoiser(self):
        self.return_torsion_noise = False
        self.torsion_edge_embedder = None
        self.final_tp_tor = None
        self.tor_bond_conv = None
        self.tor_final_layer = None

    def forward(self, node_attr, coords, batch_id, perturb_mask, edges, edge_type_attr, tor_edges=None, tor_batch=None):
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
            assert not torch.any(torch.isnan(edge_length_embedding)), "nans in edge_length_embedding"
            assert not torch.any(torch.isnan(edge_type_attr)), "nans in edge_type_attr"
            assert not torch.any(torch.isnan(node_attr)), "nans in node_attr"

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
        
        if any([self.return_atom_noise, self.return_torsion_noise, self.return_global_noise]):
            if self.return_atom_noise:
                # Local denoising
                local_edge_length_embedding = self.local_denoise_edge_embedder(edge_length)
                edge_attr = torch.cat(
                    (
                        local_edge_length_embedding,
                        edge_type_attr,
                        node_attr[edges[0], : self.ns],
                        node_attr[edges[1], : self.ns],
                    ),
                    dim=1,
                )
                pred = self.local_denoise_predictor(
                    node_attr, edges, edge_attr, edge_sh,
                )
                atom_noise = pred[:, :3] + pred[:, 3:]
            else:
                atom_noise = None
            
            if self.return_global_noise:
                # Global denoising
                center = scatter_mean(coords[perturb_mask], batch_id[perturb_mask], dim=0)
                num_centers = center.shape[0]
                global_edges = torch.stack((batch_id[perturb_mask], torch.nonzero(perturb_mask).flatten()), dim=0)
                global_edge_length = torch.norm(coords[global_edges[1]] - center[global_edges[0]], dim=-1)
                # print("global_edge_length", global_edge_length.mean(), global_edge_length.min(), global_edge_length.max(), global_edge_length.std())
                global_edge_length_embedding = self.global_denoise_edge_embedder(global_edge_length)
                global_edge_attr = torch.cat(
                    (
                        global_edge_length_embedding,
                        node_attr[global_edges[1], : self.ns],
                    ),
                    dim=1,
                )
                global_edge_sh = o3.spherical_harmonics(
                    self.sh_irreps,
                    coords[global_edges[1]] - center[global_edges[0]],
                    normalize=True,
                    normalization="component",
                )
                global_pred = self.global_denoise_predictor(
                    node_attr, global_edges, global_edge_attr, global_edge_sh, out_nodes = num_centers,
                )
                trans_noise = global_pred[:, :3] + global_pred[:, 6:9]
                rot_noise = global_pred[:, 3:6] + global_pred[:, 9:]
            else:
                trans_noise = None
                rot_noise = None

            if self.return_torsion_noise:
                if tor_edges.numel() == 0:
                    torsion_noise = None
                else:
                    assert tor_edges is not None, "Torsion edges must be provided if return_torsion_noise is True."
                
                    #  Torsion denoising
                    tor_edge_index, tor_edge_attr, tor_edge_sh = self.build_tor_edges(
                        tor_edges, coords, node_embeddings, batch_id, tor_batch)
                    pred = self.tor_bond_conv(
                        node_attr, tor_edge_index, tor_edge_attr, tor_edge_sh, out_nodes=tor_edges.shape[1],
                    )
                    torsion_noise = self.tor_final_layer(pred).flatten()
                    assert torsion_noise.shape[0] == tor_edges.shape[1], f"Torsion noise {torsion_noise.shape} must be predicted for each torsion edge {tor_edges.shape}."
            else:
                torsion_noise = None
            return self.out_ffn(node_embeddings), trans_noise, rot_noise, atom_noise, torsion_noise
        else:
            node_embeddings = self.out_ffn(node_embeddings)
            return node_embeddings

    def build_tor_edges(self, tor_bonds, coords, node_embeddings, batch_id, tor_batch):
        bond_pos = (coords[tor_bonds[1]] + coords[tor_bonds[0]]) / 2
        tor_bond_attr = node_embeddings[tor_bonds[0]] + node_embeddings[tor_bonds[1]]
        
        edge_index = radius(coords, bond_pos, self.tor_max_radius, batch_x=batch_id, batch_y=tor_batch, 
                                max_num_neighbors=self.tor_max_neighbors)
        # 0-row is torsion edge index, 1-row is the node index
        edge_vec = coords[edge_index[1]] - bond_pos[edge_index[0]]
        edge_embed = self.torsion_edge_embedder(edge_vec.norm(dim=-1))
        edge_attr = torch.cat((edge_embed,
                                node_embeddings[edge_index[1], : self.ns], 
                                tor_bond_attr[edge_index[0], : self.ns]), 
                                dim=-1)
        
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization="component")

        tor_bonds_vec = coords[tor_bonds[1]] - coords[tor_bonds[0]]
        tor_bonds_sh = o3.spherical_harmonics("2e", tor_bonds_vec, normalize=True, normalization="component")

        
        tor_edge_sh = self.final_tp_tor(edge_sh, tor_bonds_sh[edge_index[0]])
        return edge_index, edge_attr, tor_edge_sh
