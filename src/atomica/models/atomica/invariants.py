"""Rotation-invariant descriptors from ATOMICA's SE(3)-equivariant irrep tensors.

``InteractionModule`` emits per node a direct sum of irreps, ``irrep_seq[-1]`` in atomica.py:

    {ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o

with ``ns = hidden_size`` and ``nv = ns // 2``. The model's own readout keeps the 0e and 0o
channels and drops the rest, so it is invariant but carries no l>0 geometry. This module builds
descriptors that turn those channels into invariants an ordinary head can use:

    scalars      the raw 0e/0o channels (2*ns)
    norms        per-channel L2 norm of each l>0 irrep (4*nv)
    gram         within-degree inner products, upper triangle, 4 * nv(nv+1)/2 features; the
                 complete degree-2 invariant set, and it subsumes norms
    bispectrum   order-3 Clebsch-Gordan contractions, not part of the default descriptor

Degrees are never mixed outside ``bispectrum``, since cross-degree inner products are not
invariant without Clebsch-Gordan coupling. Translation invariance comes from ATOMICA building its
messages out of relative edge vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from e3nn import o3


def default_irreps_str(ns: int, nv: int) -> str:
    """The final post-convolution irrep layout (``irrep_seq[-1]`` in ``atomica.py``)."""
    return f"{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o"


class IrrepInvariants:
    """Turns an ATOMICA irrep tensor into rotation-invariant descriptors.

    Prefer the constructors that read the layout off the model (:meth:`from_hidden_size`,
    :meth:`from_encoder`) over passing ``ns``/``nv`` by hand -- a mismatched layout silently produces
    garbage rather than an error.
    """

    def __init__(self, ns: int, nv: int, irreps_str: Optional[str] = None):
        self.ns, self.nv = ns, nv
        self.irreps_str = irreps_str or default_irreps_str(ns, nv)
        self.irreps = o3.Irreps(self.irreps_str)

        # per irrep group: flat-column slice, multiplicity, (2l+1). e3nn guarantees slices() matches
        # the column layout of a tensor in this basis.
        self._scalar_slices: List[Tuple[slice, str]] = []
        self._higher: List[Dict] = []
        for (mul, ir), sl in zip(self.irreps, self.irreps.slices()):
            tag = f"{ir.l}{'e' if ir.p == 1 else 'o'}"
            if ir.l == 0:
                self._scalar_slices.append((sl, tag))
            else:
                iu, ju = torch.triu_indices(mul, mul, offset=0)
                self._higher.append(dict(slice=sl, mul=mul, dim=2 * ir.l + 1, l=ir.l, tag=tag, iu=iu, ju=ju))
        self._build_bispectrum_couplings()

    # ---- constructors that derive the layout from the model -------------------------------
    @classmethod
    def from_hidden_size(cls, hidden_size: int) -> "IrrepInvariants":
        """``ATOMICAEncoder`` builds ``InteractionModule(ns=hidden_size, nv=hidden_size//2)``."""
        return cls(ns=hidden_size, nv=hidden_size // 2)

    @classmethod
    def from_encoder(cls, encoder) -> "IrrepInvariants":
        """From an ``ATOMICAEncoder`` (or anything exposing ``.encoder.ns`` / ``.encoder.nv``)."""
        inner = getattr(encoder, "encoder", encoder)
        return cls(ns=inner.ns, nv=inner.nv)

    # ---- dimensionalities -----------------------------------------------------------------
    @property
    def n_scalars(self) -> int:
        return sum(sl.stop - sl.start for sl, _ in self._scalar_slices)      # 2*ns

    @property
    def n_norms(self) -> int:
        return sum(g["mul"] for g in self._higher)                            # 4*nv

    @property
    def n_gram(self) -> int:
        return sum(g["iu"].numel() for g in self._higher)                     # 4 * nv(nv+1)/2

    @property
    def n_bispectrum(self) -> int:
        if not self._bisp_ok or not self._bisp_couplings:
            return 0
        return len(self._bisp_couplings) * self._higher[0]["mul"]

    @property
    def n_descriptor(self) -> int:
        """Width of :meth:`descriptor` (scalars + gram)."""
        return self.n_scalars + self.n_gram

    def check_dim(self, node_attr: torch.Tensor) -> None:
        if node_attr.shape[-1] != self.irreps.dim:
            raise ValueError(
                f"node_attr last dim {node_attr.shape[-1]} != irreps.dim {self.irreps.dim} for "
                f"'{self.irreps_str}'. Build this object from the model (from_hidden_size/from_encoder)."
            )

    # ---- invariants -----------------------------------------------------------------------
    def scalars(self, node_attr: torch.Tensor) -> torch.Tensor:
        """The raw 0e and 0o channels (2*ns). Already invariant."""
        return torch.cat([node_attr[:, sl] for sl, _ in self._scalar_slices], dim=-1)

    def norms(self, node_attr: torch.Tensor) -> torch.Tensor:
        """Per-channel L2 norm of each l>0 irrep (4*nv)."""
        out = []
        for g in self._higher:
            x = node_attr[:, g["slice"]].reshape(node_attr.shape[0], g["mul"], g["dim"])
            out.append(x.norm(dim=-1))
        return torch.cat(out, dim=-1)

    def gram(self, node_attr: torch.Tensor) -> torch.Tensor:
        """Within-degree Gram upper triangle (incl. diagonal). Subsumes :meth:`norms`."""
        out = []
        for g in self._higher:
            x = node_attr[:, g["slice"]].reshape(node_attr.shape[0], g["mul"], g["dim"])
            gram = torch.einsum("nmd,npd->nmp", x, x)
            out.append(gram[:, g["iu"], g["ju"]])
        return torch.cat(out, dim=-1)

    def descriptor(self, node_attr: torch.Tensor) -> torch.Tensor:
        """The default per-node invariant descriptor: scalars (+) within-degree Gram."""
        return torch.cat([self.scalars(node_attr), self.gram(node_attr)], dim=-1)

    # ---- order-3 (not part of the default descriptor) -------------------------------------
    def _build_bispectrum_couplings(self):
        """Channel-diagonal cross-degree CG couplings, as used in SOAP/ACE-style descriptors.

        b = sum_ijk C^{la lb lc}_{ijk} x^{(la)}_i x^{(lb)}_j x^{(lc)}_k  is SO(3)-invariant because the
        Wigner 3j tensor is an intertwiner. We enumerate unordered pairs (a<=b) coupled to every l>0 group
        satisfying the triangle rule, then drop couplings whose channel-diagonal contraction is
        identically zero (e.g. antisymmetric x cross x cases). Pruning uses a fixed seeded probe, so the
        kept set is deterministic across processes and checkpoints.
        """
        groups = self._higher
        muls = {g["mul"] for g in groups}
        self._bisp_ok = len(muls) == 1 and len(groups) > 0
        self._bisp_couplings: List[Dict] = []
        if not self._bisp_ok:
            return
        mul = groups[0]["mul"]
        gen = torch.Generator().manual_seed(0)
        cal = torch.randn(64, self.irreps.dim, generator=gen)
        ll = [g["l"] for g in groups]
        for a in range(len(groups)):
            for b in range(a, len(groups)):
                for c in range(len(groups)):
                    la, lb, lc = ll[a], ll[b], ll[c]
                    if not (abs(la - lb) <= lc <= la + lb):
                        continue
                    coup = dict(a=a, b=b, c=c, W=o3.wigner_3j(la, lb, lc))
                    if self._one_bispectrum(cal, coup, mul).abs().max() > 1e-6:
                        self._bisp_couplings.append(coup)

    def _one_bispectrum(self, node_attr, coup, mul):
        ga, gb, gc = self._higher[coup["a"]], self._higher[coup["b"]], self._higher[coup["c"]]
        n = node_attr.shape[0]
        Xa = node_attr[:, ga["slice"]].reshape(n, mul, ga["dim"])
        Xb = node_attr[:, gb["slice"]].reshape(n, mul, gb["dim"])
        Xc = node_attr[:, gc["slice"]].reshape(n, mul, gc["dim"])
        W = coup["W"].to(node_attr.dtype).to(node_attr.device)
        return torch.einsum("ijk,nci,ncj,nck->nc", W, Xa, Xb, Xc)

    def bispectrum(self, node_attr: torch.Tensor) -> torch.Tensor:
        """Channel-diagonal order-3 invariants. Not part of the default descriptor."""
        if not self._bisp_ok or not self._bisp_couplings:
            return node_attr.new_zeros((node_attr.shape[0], 0))
        mul = self._higher[0]["mul"]
        return torch.cat([self._one_bispectrum(node_attr, c, mul) for c in self._bisp_couplings], dim=-1)


def pool_atoms_to_blocks(x: torch.Tensor, block_id: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Pool per-atom invariants into per-block features by mean and standard deviation.

    The standard deviation carries how heterogeneous a block's atomic environments are, which the
    mean discards. ``block_id`` is a global atom-to-block index, so this works batched. Returns
    ``[n_blocks, 2*d]``.
    """
    from ...utils.scatter import scatter_mean
    mean = scatter_mean(x, block_id, dim=0, dim_size=n_blocks)
    mean_sq = scatter_mean(x * x, block_id, dim=0, dim_size=n_blocks)
    std = (mean_sq - mean * mean).clamp_min(0).sqrt()
    return torch.cat([mean, std], dim=-1)
