"""Step 2: rank the pockets by cosine and report the retrieval statistics.

Six statistics rather than one, because they differ in how fast their weight decays with rank:
AUROC is uniform over all ranks, nDCG is mildly rank-discounted, mAP goes roughly as 1/rank, MRR
sees only the first hit. Together they show where in the ranking an advantage lives.

Usage:
    python evaluate_retrieval.py
    python evaluate_retrieval.py --vectors "my method=path/to/vectors.npz"
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import benchmark as B

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DEFAULT_VECTORS = f"ATOMICA (frozen)={os.path.join(HERE, 'representations', 'atomica_z_graph_cn.npz')}"


def describe_pool(bench: dict, layout: dict) -> None:
    pockets = bench["pockets"]
    queries = list(layout)
    n_relevant = [int(L["relevant"].sum()) for L in layout.values()]
    n_ranked = [len(L["relevant"]) for L in layout.values()]
    print(f"{len(pockets)} pockets, {len(queries)} queries, "
          f"{len({pockets[q]['ligand'] for q in queries})} ligand classes, "
          f"{len({pockets[q]['cluster'] for q in queries})} sequence clusters")
    print(f"{sum(n_relevant) // 2} positive pairs; median {int(np.median(n_relevant))} relevant "
          f"and {int(np.median(n_ranked))} candidates per query")
    print(f"positive prevalence {np.mean(n_relevant) / np.mean(n_ranked) * 100:.2f}%\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--vectors", action="append", default=None, metavar="LABEL=PATH",
                        help="repeat to score several representations")
    parser.add_argument("--benchmark", default=os.path.join(B.DATA, "benchmark.json"))
    parser.add_argument("--out", default=os.path.join(RESULTS, "retrieval_metrics.json"))
    parser.add_argument("--n_boot", type=int, default=B.N_BOOT)
    parser.add_argument("--n_perm", type=int, default=B.N_PERM)
    args = parser.parse_args()

    entries = args.vectors or [DEFAULT_VECTORS]
    bench = B.load_benchmark(args.benchmark)

    first_ids, _ = B.load_vectors(entries[0].split("=", 1)[1])
    layout = B.query_layout(first_ids, bench)
    describe_pool(bench, layout)

    print(f"chance value of each statistic, by permuting the ranking "
          f"({args.n_perm} draws)...", flush=True)
    chance, _ = B.chance_table(layout, bench, n_perm=args.n_perm, seed=B.SEED)

    rows = {}
    for entry in entries:
        label, path = entry.split("=", 1)
        ids, matrix = B.load_vectors(path)
        if ids != first_ids:
            raise SystemExit(f"{label}: pocket order differs from {entries[0].split('=')[0]}")
        scores = B.score_all_queries(matrix, ids, bench, layout)
        rows[label] = B.summarise(scores, bench, chance, by="cluster", n_boot=args.n_boot)
        rows[label]["dim"] = int(matrix.shape[1])

    print(f"\nmacro-averaged over sequence clusters, 95% bootstrap interval over queries "
          f"({args.n_boot} resamples)\n")
    print(B.format_header())
    print(B.format_row("random reference", {k: chance[k] for k in B.METRICS}))
    for label, row in rows.items():
        print(B.format_row(label, row))
    print()
    for label, row in rows.items():
        print(f"{label}: mAP lift over chance {row['mAP_lift']:.2f}x  ({row['dim']} dimensions, "
              f"{row['n_queries']} queries)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"n_boot": args.n_boot, "n_perm": args.n_perm, "seed": B.SEED,
                   "macro_by": "cluster", "chance": chance, "results": rows}, handle, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
