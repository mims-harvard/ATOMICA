"""
Verify that ATOMICA's invariant representations really are SE(3)-invariant -- measured, not asserted.

Takes a real structure, runs the frozen backbone, then applies a random proper rotation + translation to
the input coordinates and runs it again. Checks:

  * every ``*_scalar_repr`` and every ``*_invariant_repr`` is unchanged, AND
  * the raw ``l = 1`` irrep components DO change.

The second check is what makes the first meaningful: without it, a bug that zeroed the features would
"pass". Also verifies that `return_invariant_repr=True` does not perturb the standard readout, which is
the backward-compatibility guarantee.

Run:
    python -m atomica.probe.test_invariance --data <processed.parquet> \
        --config checkpoints/pretrain/pretrain_model_config.json \
        --weights checkpoints/pretrain/pretrain_model_weights.pt
Exit code 0 == all checks pass.
"""

from __future__ import annotations

import argparse
import sys

import torch


def random_rotation(seed: int) -> torch.Tensor:
    """Uniformly-random PROPER rotation (det=+1) via QR. ATOMICA is SE(3)- not O(3)-equivariant, so
    reflections are deliberately excluded (pseudoscalar 0o channels would flip sign under them)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(3, 3, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a)
    q = q * torch.sign(torch.diag(r))
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def run(model, batch, device):
    return model.forward(
        Z=batch["X"].to(device), B=batch["B"].to(device), A=batch["A"].to(device),
        block_lengths=batch["block_lengths"].to(device), lengths=batch["lengths"].to(device),
        segment_ids=batch["segment_ids"].to(device),
        return_graph_repr=True, return_invariant_repr=True,
    )


def main():
    from atomica.data.dataset import MultiClassLabelledPDBDataset
    from atomica.models.prediction_model import PredictionModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="a *_processed.parquet readable by the dataset class")
    ap.add_argument("--config", default="checkpoints/pretrain/pretrain_model_config.json")
    ap.add_argument("--weights", default="checkpoints/pretrain/pretrain_model_weights.pt")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-2, help="relative tolerance; higher-order invariants "
                    "have larger magnitudes so their float32 error is larger in ABSOLUTE terms")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = PredictionModel.load_from_config_and_weights(args.config, args.weights).to(args.device).eval()
    ds = MultiClassLabelledPDBDataset(args.data)
    batch = MultiClassLabelledPDBDataset.collate_fn([ds[0]])

    R = random_rotation(args.seed)
    t = torch.randn(3, dtype=torch.float64,
                    generator=torch.Generator().manual_seed(args.seed + 1)) * 10.0

    with torch.no_grad():
        out0 = run(model, batch, args.device)
        rot = dict(batch)
        rot["X"] = ((batch["X"].double() @ R.T) + t).to(batch["X"].dtype)
        out1 = run(model, rot, args.device)
        # the new flag must not perturb the standard readout. CUDA scatter reductions are not
        # bitwise-deterministic, so compare the flag-induced difference against a same-input
        # run-to-run baseline rather than demanding exact equality.
        def _plain():
            return model.forward(
                Z=batch["X"].to(args.device), B=batch["B"].to(args.device), A=batch["A"].to(args.device),
                block_lengths=batch["block_lengths"].to(args.device),
                lengths=batch["lengths"].to(args.device),
                segment_ids=batch["segment_ids"].to(args.device))
        plain, plain2 = _plain(), _plain()

    checks = {
        "atom_scalar_repr":      (out0.atom_scalar_repr, out1.atom_scalar_repr),
        "block_scalar_repr":     (out0.block_scalar_repr, out1.block_scalar_repr),
        "graph_scalar_repr":     (out0.graph_scalar_repr, out1.graph_scalar_repr),
        "atom_invariant_repr":   (out0.atom_invariant_repr, out1.atom_invariant_repr),
        "block_invariant_repr":  (out0.block_invariant_repr, out1.block_invariant_repr),
        "graph_invariant_repr":  (out0.graph_invariant_repr, out1.graph_invariant_repr),
    }

    ok = True
    print(f"{'representation':24} {'shape':>18} {'max|d|':>11} {'max rel d':>11}   result")
    print("-" * 82)
    for name, (a, b) in checks.items():
        a, b = a.double(), b.double()
        mabs = (a - b).abs().max().item()
        mrel = ((a - b).abs() / a.abs().max().clamp_min(1e-6)).max().item()
        passed = (mabs <= args.atol) or (mrel <= args.rtol)
        ok &= passed
        print(f"{name:24} {str(tuple(a.shape)):>18} {mabs:>11.2e} {mrel:>11.2e}   "
              f"{'PASS' if passed else 'FAIL'}")

    # sanity: raw l=1 components MUST rotate, else the test above is vacuous
    ns = model.top_encoder.encoder.ns
    nv = model.top_encoder.encoder.nv
    one_o0 = out0.block_node_attr[:, ns:ns + 3 * nv].double()
    one_o1 = out1.block_node_attr[:, ns:ns + 3 * nv].double()
    changed = (one_o0 - one_o1).abs().max().item()
    sane = changed > 1e-2
    print("-" * 82)
    print(f"{'raw l=1 (must CHANGE)':24} {str(tuple(one_o0.shape)):>18} {changed:>11.2e} "
          f"{'':>11}   {'PASS' if sane else 'FAIL'}")

    # backward compatibility: the flag must not alter the standard readout by more than the model's own
    # run-to-run non-determinism on identical input.
    d = lambda a, b: (a.double() - b.double()).abs().max().item()
    base_block, base_graph = d(plain.block_repr, plain2.block_repr), d(plain.graph_repr, plain2.graph_repr)
    flag_block, flag_graph = d(plain.block_repr, out0.block_repr), d(plain.graph_repr, out0.graph_repr)
    tol = max(1e-6, 10 * max(base_block, base_graph))
    compat = (flag_block <= tol) and (flag_graph <= tol)
    print("-" * 82)
    print(f"run-to-run non-determinism (same input):  block={base_block:.2e}  graph={base_graph:.2e}")
    print(f"flag-induced difference:                  block={flag_block:.2e}  graph={flag_graph:.2e}"
          f"   (tol {tol:.1e})   {'PASS' if compat else 'FAIL'}")

    all_ok = ok and sane and compat
    print("\n" + ("ALL CHECKS PASSED" if all_ok else "CHECKS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
