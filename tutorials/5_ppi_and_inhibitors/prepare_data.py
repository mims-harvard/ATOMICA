"""
prepare_data.py — regenerate every parquet/csv in data/ from raw 2P2IDB
structures and the pretrained ATOMICA checkpoint.

Inputs you need locally:
  * A directory containing 2P2IDB mmCIF (and/or PDB) files named
    <PDB>.cif (e.g. 1YSW.cif). Download with gh/rsync from 2P2IDB or PDB.
  * A pretrained ATOMICA checkpoint (see tutorial README).
  * MSMS binary on PATH (or $MSMS_BIN set).

Outputs (all under data/):
  - inhibitors_index.csv                          (data_index_file for process_pdbs)
  - inhibitors_processed.parquet
  - inhibitors_embeddings.parquet
  - peptide_partners_processed.parquet
  - peptide_partners_embeddings.parquet
  - protein_partner_surface_patches.parquet
  - protein_partner_surface_patches_embeddings.parquet
  - protein_partner_surface_patches_distances.csv
  - peptide_inhibitor_block_results.parquet

Usage:
    python prepare_data.py --cif_dir /path/to/2p2idb/cifs --ckpt_dir /path/to/ATOMICA_checkpoints/pretrain
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import biotite.sequence as seq
import biotite.sequence.align as align
from biotite.sequence.align import SubstitutionMatrix
import scipy.spatial.distance

from atomica.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from atomica.data.dataset import blocks_to_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_sampler import get_mesh_and_sample

DATA_DIR = Path(__file__).resolve().parent / "data"


# ---------- structure helpers ----------

def _load_structure_any(path: str, model: int = 1):
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".pdb", ".ent"}:
        return pdb.PDBFile.read(p).get_structure(model=model)
    if suf in {".cif", ".mmcif"}:
        return pdbx.get_structure(pdbx.PDBxFile.read(p), model=model)
    raise ValueError(f"Unsupported file type: {suf}")


def resolve_structure_path(cif_dir: Path, pdb_id: str) -> Path | None:
    for ext in (".cif", ".pdb"):
        p = cif_dir / f"{pdb_id}{ext}"
        if p.exists():
            return p
    return None


def check_chain_exists(path: Path, chain: str):
    arr = _load_structure_any(path)
    uniq = np.unique(arr.chain_id).tolist()
    return chain in uniq, uniq


def get_lig_resi(path: Path, chain: str, lig_code: str) -> List[int]:
    arr = _load_structure_any(path)
    mask = (arr.chain_id == chain) & (arr.res_name == lig_code.strip().upper())
    if not np.any(mask):
        return []
    return sorted(np.unique(arr.res_id[mask]).tolist())


def infill_chain1(row):
    chain3 = row["chain1"] * 3
    if len(row["chain1"]) == 1 and chain3 in row["unique_chains"]:
        return chain3
    if row["chain2"] in row["unique_chains"]:
        return row["chain2"]
    if len(row["unique_chains"]) == 1:
        return row["unique_chains"][0]
    return None


# ---------- step 1: build inhibitors_index.csv for atomica.data.process_pdbs ----------

def build_inhibitors_index(cif_dir: Path) -> pd.DataFrame:
    """Add a pdb_path column pointing into cif_dir and explode one row per ligand resi."""
    meta = pd.read_csv(DATA_DIR / "inhibitors_metadata.csv")

    meta["pdb_path"] = meta["pdb_code"].apply(
        lambda x: str(resolve_structure_path(cif_dir, x)) if resolve_structure_path(cif_dir, x) else None
    )
    missing = meta["pdb_path"].isna().sum()
    if missing:
        print(f"[inhibitors] {missing} / {len(meta)} PDB codes missing under {cif_dir} — dropping")
        meta = meta[meta["pdb_path"].notna()].copy()

    chain_info = meta.apply(
        lambda r: pd.Series(check_chain_exists(Path(r["pdb_path"]), r["chain1"]),
                            index=["chain1_exists", "unique_chains"]),
        axis=1,
    )
    meta[["chain1_exists", "unique_chains"]] = chain_info

    fix_mask = ~meta["chain1_exists"]
    if fix_mask.any():
        meta.loc[fix_mask, "chain1"] = meta[fix_mask].apply(infill_chain1, axis=1)
        meta = meta[meta["chain1"].notna()].copy()

    meta["lig_resi"] = meta.apply(
        lambda r: get_lig_resi(Path(r["pdb_path"]), r["chain2"], r["lig_code"]), axis=1
    )
    meta = meta[meta["lig_resi"].apply(len) > 0].explode("lig_resi")
    meta = meta.drop(columns=["chain1_exists", "unique_chains"])

    out = DATA_DIR / "inhibitors_index.csv"
    meta.to_csv(out, index=False)
    print(f"[inhibitors] wrote {out} with {len(meta)} rows")
    return meta


# ---------- step 2: peptide partner blocks (≤30 residues) ----------

def build_peptide_partners(cif_dir: Path):
    mapping = pd.read_csv(DATA_DIR / "ppi_inhibitor_mapping.csv")
    ppis = (mapping[["PDBProtProt", "Family", "Chain_Target", "Chain_Partner"]]
            .drop_duplicates()
            .query("PDBProtProt != 'na'")
            .reset_index(drop=True))

    rows = []
    for _, row in tqdm(ppis.iterrows(), total=len(ppis), desc="Partner chain blocks"):
        path = resolve_structure_path(cif_dir, row["PDBProtProt"])
        if path is None:
            continue
        try:
            blocks, pdb_idx = pdb_to_list_blocks(str(path), [row["Chain_Partner"]], return_indexes=True)
        except Exception as e:
            print(f"  skip {row['PDBProtProt']}_{row['Chain_Partner']}: {e}")
            continue
        blocks = sum(blocks, [])
        pdb_idx = sum(pdb_idx, [])
        if len(blocks) == 0 or len(blocks) > 30:
            continue  # protein partners go through the surface-patch path
        data = blocks_to_data(blocks)
        data["id"] = f"{row['PDBProtProt']}_{row['Chain_Partner']}"
        data["block_to_pdb_indexes"] = json.dumps(
            {k: v for k, v in zip(range(1, len(blocks) + 1), pdb_idx)}
        )
        rows.append(data)

    out = DATA_DIR / "peptide_partners_processed.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"[peptide] wrote {out} with {len(rows)} rows")


# ---------- step 3: protein partner surface patches (>30 residues) ----------

def build_protein_partner_patches(cif_dir: Path, mesh_dir: Path,
                                   num_points: int = 1000,
                                   interface_radius: float = 16.0,
                                   min_blocks_per_point: int = 8):
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(DATA_DIR / "ppi_inhibitor_mapping.csv")
    ppis = (mapping[["PDBProtProt", "Family", "Chain_Target", "Chain_Partner"]]
            .drop_duplicates()
            .query("PDBProtProt != 'na'")
            .reset_index(drop=True))

    rows = []
    dist_rows = []
    for _, row in tqdm(ppis.iterrows(), total=len(ppis), desc="Protein partner patches"):
        path = resolve_structure_path(cif_dir, row["PDBProtProt"])
        if path is None:
            continue
        try:
            blocks, pdb_idx = pdb_to_list_blocks(str(path), [row["Chain_Partner"]], return_indexes=True)
        except Exception as e:
            print(f"  skip {row['PDBProtProt']}_{row['Chain_Partner']}: {e}")
            continue
        blocks = sum(blocks, [])
        pdb_idx = sum(pdb_idx, [])
        if len(blocks) <= 30:
            continue  # peptides handled elsewhere

        mesh_ply = mesh_dir / f"{row['PDBProtProt']}_chain{row['Chain_Partner']}_mesh.ply"
        points_xyz = mesh_dir / f"{row['PDBProtProt']}_chain{row['Chain_Partner']}_points.xyz"
        if not points_xyz.exists():
            get_mesh_and_sample(
                str(path), row["Chain_Partner"], num_points=num_points,
                mesh_output_path=str(mesh_ply),
                points_output_path=str(points_xyz),
                seed=42,
            )

        points = pd.read_csv(points_xyz, sep=" ", header=None, skiprows=2)
        block_coords = np.array([b.coords for b in blocks])

        for pidx, pt in points.iterrows():
            _, x, y, z = pt
            d = np.linalg.norm(block_coords - np.array([x, y, z]), axis=1)
            mask = d < interface_radius
            if mask.sum() < min_blocks_per_point:
                continue
            near_blocks = [b for b, m in zip(blocks, mask) if m]
            near_pdb = [p for p, m in zip(pdb_idx, mask) if m]
            data = blocks_to_data(near_blocks)
            data["block_to_pdb_indexes"] = json.dumps(
                {k: v for k, v in zip(range(1, len(near_blocks) + 1), near_pdb)}
            )
            patch_id = f"{row['PDBProtProt']}_{row['Chain_Partner']}_{pidx}"
            data["id"] = patch_id
            rows.append(data)

            resi, resn, dist = closest_residue_on_chain(
                str(path), np.array([x, y, z]), row["Chain_Target"], atom_filter="ca"
            )
            dist_rows.append({"id": patch_id, "resi": resi, "resn": resn, "distance": dist})

    out = DATA_DIR / "protein_partner_surface_patches.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"[protein] wrote {out} with {len(rows)} rows")

    dist_out = DATA_DIR / "protein_partner_surface_patches_distances.csv"
    pd.DataFrame(dist_rows).to_csv(dist_out, index=False)
    print(f"[protein] wrote {dist_out}")


def closest_residue_on_chain(structure_path, xyz, chain, atom_filter="ca"):
    arr = _load_structure_any(structure_path)
    arr = arr[arr.chain_id == chain]
    if atom_filter == "ca":
        arr = arr[arr.atom_name == "CA"]
    elif atom_filter == "heavy":
        arr = arr[arr.element != "H"] if hasattr(arr, "element") else arr
    d = np.linalg.norm(arr.coord - np.asarray(xyz, float)[None, :], axis=1)
    i = int(np.argmin(d))
    return int(arr.res_id[i]), str(arr.res_name[i]), float(d[i])


# ---------- step 4: call atomica.data.process_pdbs + atomica.get_embeddings ----------

def run(cmd: List[str]):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def process_and_embed(ckpt_dir: Path):
    cfg = ckpt_dir / "pretrain_model_config.json"
    wts = ckpt_dir / "pretrain_model_weights.pt"

    # Inhibitors: fragment small molecule with PS_300
    run([
        sys.executable, "-m", "atomica.data.process_pdbs",
        "--data_index_file", str(DATA_DIR / "inhibitors_index.csv"),
        "--out_path", str(DATA_DIR / "inhibitors_processed.parquet"),
        "--fragmentation_method", "PS_300",
    ])

    pairs = [
        ("inhibitors_processed.parquet",                    "inhibitors_embeddings.parquet"),
        ("peptide_partners_processed.parquet",              "peptide_partners_embeddings.parquet"),
        ("protein_partner_surface_patches.parquet",         "protein_partner_surface_patches_embeddings.parquet"),
    ]
    for in_, out_ in pairs:
        run([
            sys.executable, "-m", "atomica.get_embeddings",
            "--model_config", str(cfg),
            "--model_weights", str(wts),
            "--data_path", str(DATA_DIR / in_),
            "--output_path", str(DATA_DIR / out_),
            "--batch_size", "8",
        ])


# ---------- step 5: peptide inhibitor block-distance matrices ----------

def _chain_ca(arr, chain):
    a = arr[(arr.chain_id == chain) & struc.filter_amino_acids(arr) & (arr.atom_name == "CA")]
    return a


def _seq_resids(ca):
    letters = []
    for r3 in ca.res_name.astype(str):
        try:
            letters.append(seq.ProteinSequence.convert_letter_3to1(r3))
        except Exception:
            letters.append("X")
    return seq.ProteinSequence("".join(letters)), ca.res_id.copy()


def _kabsch_refine(P, Q, cutoff=2.0, cycles=5):
    inliers = np.ones(P.shape[0], dtype=bool)
    R = np.eye(3); t = np.zeros(3); fitted = P.copy()
    for _ in range(cycles):
        _, tf = struc.superimpose(Q[inliers], P[inliers])
        fitted = tf.apply(P)
        d = np.linalg.norm(fitted - Q, axis=1)
        new_in = d <= cutoff
        if np.array_equal(new_in, inliers):
            R = tf.rotation[0]
            t = tf.target_translation[0] + tf.center_translation[0] @ R
            break
        if new_in.sum() < 3:
            _, tf = struc.superimpose(Q[inliers], P[inliers])
            fitted = tf.apply(P)
            R = tf.rotation[0]
            t = tf.target_translation[0] + tf.center_translation[0] @ R
            break
        inliers = new_in
        R = tf.rotation[0]
        t = tf.target_translation[0] + tf.center_translation[0] @ R
    return R, t


def align_chains(pdb1, chain1, pdb2, chain2):
    a1 = _load_structure_any(pdb1); a2 = _load_structure_any(pdb2)
    ca1 = _chain_ca(a1, chain1); ca2 = _chain_ca(a2, chain2)
    s1, _ = _seq_resids(ca1); s2, _ = _seq_resids(ca2)
    ali = align.align_optimal(
        s1, s2, SubstitutionMatrix.std_protein_matrix(),
        gap_penalty=(-10, -1), local=False,
    )[0]
    idx1, idx2 = [], []
    for row in ali.trace:
        if row[0] != -1 and row[1] != -1:
            idx1.append(row[0]); idx2.append(row[1])
    P = ca1.coord[np.array(idx1)]
    Q = ca2.coord[np.array(idx2)]
    return _kabsch_refine(P, Q)


def get_block_coords(X, block_lengths):
    out, cur = [], 0
    for L in block_lengths:
        out.append(X[cur:cur + L].mean(axis=0))
        cur += L
    return np.array(out)


def build_peptide_inhibitor_block_results(cif_dir: Path):
    ppi_emb = pd.read_parquet(DATA_DIR / "peptide_partners_embeddings.parquet")
    ppi_in = pd.read_parquet(DATA_DIR / "peptide_partners_processed.parquet")
    ppi = ppi_emb.merge(ppi_in, on="id", how="left")
    mapping = pd.read_csv(DATA_DIR / "ppi_inhibitor_mapping.csv").query("PDBProtProt != 'na'")
    ppi_meta = mapping[["PDBProtProt", "Chain_Target", "Chain_Partner", "Family"]].drop_duplicates()
    ppi["pdb_id"] = ppi["id"].str.split("_").str[0]
    ppi = ppi.merge(ppi_meta, left_on="pdb_id", right_on="PDBProtProt", how="left")

    inh_emb = pd.read_parquet(DATA_DIR / "inhibitors_embeddings.parquet")
    inh_in = pd.read_parquet(DATA_DIR / "inhibitors_processed.parquet")
    inh = pd.concat([inh_in, inh_emb.drop(columns=["id"])], axis=1)
    inh["2P2IDB_ID"] = inh["id"].str.split("_").str[0]
    meta = pd.read_csv(DATA_DIR / "inhibitors_metadata.csv").rename(columns={
        "pdb_id": "2P2IDB_ID", "chain1": "Chain_Target",
        "chain2": "ChainID_Ligand", "pdb_code": "PDBProtLig",
    })
    inh = inh.merge(meta, on="2P2IDB_ID", how="left")

    families = set(inh["Family"].dropna()) & set(ppi["Family"].dropna())
    inh = inh[inh["Family"].isin(families)].reset_index(drop=True)

    rows = []
    for i, entry in tqdm(inh.iterrows(), total=len(inh), desc="block-distance matrices"):
        ppi_row = ppi[ppi["Family"] == entry["Family"]].iloc[0]
        p_inh = resolve_structure_path(cif_dir, entry["PDBProtLig"])
        p_ppi = resolve_structure_path(cif_dir, ppi_row["pdb_id"])
        if p_inh is None or p_ppi is None:
            continue
        try:
            R, t = align_chains(str(p_inh), entry["Chain_Target"],
                                str(p_ppi), ppi_row["Chain_Target"])
        except Exception as e:
            print(f"  align fail {entry['PDBProtLig']} -> {ppi_row['pdb_id']}: {e}")
            continue

        ppi_coords = np.stack(ppi_row["X"])
        ppi_bl = ppi_row["block_lengths"]
        ppi_bc = get_block_coords(ppi_coords, ppi_bl)
        ppi_be = np.stack(ppi_row["block_embedding"])

        seg = entry["segment_ids"]
        n_pocket = entry["block_lengths"][seg == 0].sum()
        inh_coords = np.stack(entry["X"])[n_pocket:]
        inh_bl = entry["block_lengths"][seg == 1]
        if (inh_bl > 1).sum() == 0:
            continue
        inh_coords = inh_coords @ R.T + t
        inh_bc = get_block_coords(inh_coords, inh_bl)
        inh_be = np.stack(entry["block_embedding"][seg == 1])

        ed = scipy.spatial.distance.cdist(inh_be, ppi_be, metric="cosine")
        cd = scipy.spatial.distance.cdist(inh_bc, ppi_bc)
        ed = ed[inh_bl > 1, :][:, ppi_bl > 1]
        cd = cd[inh_bl > 1, :][:, ppi_bl > 1]

        rows.append({
            "ppi_pdb_id": ppi_row["pdb_id"],
            "ppi_chain_target": ppi_row["Chain_Target"],
            "ppi_chain_partner": ppi_row["Chain_Partner"],
            "inhibitor_pdb_id": entry["PDBProtLig"],
            "inhibitor_chain_target": entry["Chain_Target"],
            "family": entry["Family"],
            "lig_code": entry["id"].split("_")[-1],
            "inhibitor_index": i,
            "min_dist": float(cd.min()),
            "block_emb_dist": ed.flatten(),
            "block_coords_dist": cd.flatten(),
            "shape_block_emb_dist": ed.shape,
            "shape_block_coords_dist": cd.shape,
        })

    out = DATA_DIR / "peptide_inhibitor_block_results.parquet"
    pd.DataFrame(rows).to_parquet(out)
    print(f"[block-results] wrote {out} with {len(rows)} rows")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cif_dir", required=True, help="Directory with 2P2IDB <PDB>.cif (or .pdb) files")
    p.add_argument("--ckpt_dir", default="checkpoints/ATOMICA_checkpoints/pretrain",
                   help="Pretrained ATOMICA checkpoint directory")
    p.add_argument("--mesh_dir", default=str(DATA_DIR / "surface_mesh"),
                   help="Cache directory for MSMS mesh + sampled points")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["index", "peptide", "protein", "embed", "blockdist"])
    args = p.parse_args()

    cif_dir = Path(args.cif_dir)
    ckpt_dir = Path(args.ckpt_dir)
    mesh_dir = Path(args.mesh_dir)
    assert cif_dir.is_dir(), cif_dir

    if "index" not in args.skip:
        build_inhibitors_index(cif_dir)
    if "peptide" not in args.skip:
        build_peptide_partners(cif_dir)
    if "protein" not in args.skip:
        build_protein_partner_patches(cif_dir, mesh_dir)
    if "embed" not in args.skip:
        process_and_embed(ckpt_dir)
    if "blockdist" not in args.skip:
        build_peptide_inhibitor_block_results(cif_dir)


if __name__ == "__main__":
    main()
