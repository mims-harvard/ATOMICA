"""Rebuild everything in data/ from raw 2P2Idb structures.

Only needed to regenerate the deposited data. To run the tutorial, download data/ and go
straight to compute_embeddings.py.

Requires a directory of 2P2Idb mmCIF or PDB files named <PDB>.cif and the MSMS binary on
PATH or in $MSMS_BIN.

    python prepare_data.py --cif_dir /path/to/2p2idb/cifs
"""

import argparse
import json
import sys
from pathlib import Path

import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
import pandas as pd
import scipy.spatial.distance
from biotite.sequence.align import SubstitutionMatrix
from tqdm import tqdm

from atomica.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from atomica.data.dataset import blocks_to_data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surface_sampler import get_mesh_and_sample

DATA_DIR = Path(__file__).resolve().parent / "data"

MAX_PEPTIDE_RESIDUES = 30
PATCH_RADIUS = 16.0
PATCH_MIN_BLOCKS = 8
N_SURFACE_POINTS = 1000


def load_structure(path, model=1):
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return pdb.PDBFile.read(path).get_structure(model=model)
    if suffix in {".cif", ".mmcif"}:
        return pdbx.get_structure(pdbx.PDBxFile.read(path), model=model)
    raise ValueError(f"unsupported file type: {suffix}")


def find_structure(cif_dir, pdb_id):
    for ext in (".cif", ".pdb"):
        path = cif_dir / f"{pdb_id}{ext}"
        if path.exists():
            return path
    return None


def partner_blocks(cif_dir, complex_row):
    """Blocks of one complex's partner chain, or None if the structure is missing."""
    path = find_structure(cif_dir, complex_row["pdb"])
    if path is None:
        return None
    blocks, indexes = pdb_to_list_blocks(str(path), [complex_row["chain_partner"]],
                                         return_indexes=True)
    return sum(blocks, []), sum(indexes, []), path


def build_peptide_partners(cif_dir, complexes):
    """Partner chains of 30 residues or fewer, as single-graph ATOMICA inputs."""
    rows = []
    for _, complex_row in tqdm(complexes.iterrows(), total=len(complexes),
                               desc="peptide partners"):
        result = partner_blocks(cif_dir, complex_row)
        if result is None:
            continue
        blocks, indexes, _ = result
        if not blocks or len(blocks) > MAX_PEPTIDE_RESIDUES:
            continue
        data = blocks_to_data(blocks)
        data["id"] = f"{complex_row['pdb']}_{complex_row['chain_partner']}"
        data["block_to_pdb_indexes"] = json.dumps(
            dict(zip(range(1, len(blocks) + 1), indexes)))
        rows.append(data)
    out = DATA_DIR / "peptide_partners_processed.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(rows)} peptide partners")


def nearest_target_residue(path, point, chain):
    array = load_structure(path)
    array = array[(array.chain_id == chain) & (array.atom_name == "CA")]
    distances = np.linalg.norm(array.coord - np.asarray(point, float), axis=1)
    i = int(np.argmin(distances))
    return int(array.res_id[i]), str(array.res_name[i]), float(distances[i])


def build_surface_patches(cif_dir, complexes, mesh_dir):
    """Local surface patches on partner chains longer than 30 residues.

    MSMS surface at density 3.0 and probe radius 1.5 A, 1,000 area-weighted points per
    chain, each patch the blocks within 16 A of a point. Points with fewer than 8 nearby
    blocks are dropped. The distance from each patch centre to the nearest target-chain CA
    is the geometric label.
    """
    mesh_dir.mkdir(parents=True, exist_ok=True)
    patches = []
    for _, complex_row in tqdm(complexes.iterrows(), total=len(complexes),
                               desc="surface patches"):
        result = partner_blocks(cif_dir, complex_row)
        if result is None:
            continue
        blocks, indexes, path = result
        if len(blocks) <= MAX_PEPTIDE_RESIDUES:
            continue

        stem = f"{complex_row['pdb']}_chain{complex_row['chain_partner']}"
        points_file = mesh_dir / f"{stem}_points.xyz"
        if not points_file.exists():
            get_mesh_and_sample(str(path), complex_row["chain_partner"],
                                num_points=N_SURFACE_POINTS,
                                mesh_output_path=str(mesh_dir / f"{stem}_mesh.ply"),
                                points_output_path=str(points_file), seed=42)
        points = pd.read_csv(points_file, sep=" ", header=None, skiprows=2)
        centres = np.array([b.coords for b in blocks])

        for index, point in points.iterrows():
            xyz = np.asarray(point[1:4], dtype=float)
            near = np.linalg.norm(centres - xyz, axis=1) < PATCH_RADIUS
            if near.sum() < PATCH_MIN_BLOCKS:
                continue
            kept = [b for b, m in zip(blocks, near) if m]
            data = blocks_to_data(kept)
            data["id"] = f"{complex_row['pdb']}_{complex_row['chain_partner']}_{index}"
            data["block_to_pdb_indexes"] = json.dumps(dict(zip(
                range(1, len(kept) + 1), [p for p, m in zip(indexes, near) if m])))

            _, _, distance = nearest_target_residue(path, xyz,
                                                    complex_row["chain_target"])
            data["distance_to_target"] = np.float32(distance)
            patches.append(data)

    out = DATA_DIR / "surface_patches_processed.parquet"
    pd.DataFrame(patches).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(patches)} patches")


def chain_ca(array, chain, path):
    mask = array.chain_id == chain
    if not np.any(mask):
        raise ValueError(f"chain {chain!r} not in {path}")
    array = array[mask]
    array = array[struc.filter_amino_acids(array) & (array.atom_name == "CA")]
    if len(array) == 0:
        raise ValueError(f"no CA atoms on chain {chain!r} in {path}")
    return array


def chain_sequence(ca):
    letters = []
    for name in ca.res_name.astype(str):
        try:
            letters.append(seq.ProteinSequence.convert_letter_3to1(name))
        except Exception:
            letters.append("X")
    return seq.ProteinSequence("".join(letters))


def superpose_target_chains(path1, chain1, path2, chain2, cutoff=2.0, cycles=5):
    """Align two target chains by sequence, then superpose their CA atoms.

    BLOSUM62 with gap penalties -10 and -1, then iterative Kabsch refinement rejecting
    pairs beyond `cutoff`. Returns biotite's transform, its RMSD and the inlier count;
    apply it with transform.apply(coords).
    """
    ca1 = chain_ca(load_structure(path1), chain1, path1)
    ca2 = chain_ca(load_structure(path2), chain2, path2)
    alignment = align.align_optimal(
        chain_sequence(ca1), chain_sequence(ca2),
        SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1), local=False)[0]
    matched = [(a, b) for a, b in alignment.trace if a != -1 and b != -1]
    if len(matched) < 3:
        raise ValueError(f"alignment matched only {len(matched)} residues")

    P = ca1.coord[np.array([a for a, _ in matched])]
    Q = ca2.coord[np.array([b for _, b in matched])]
    inliers, transform, fitted = np.ones(len(P), bool), None, P.copy()
    for _ in range(cycles):
        _, transform = struc.superimpose(Q[inliers], P[inliers])
        fitted = transform.apply(P)
        updated = np.linalg.norm(fitted - Q, axis=1) <= cutoff
        if np.array_equal(updated, inliers) or updated.sum() < 3:
            break
        inliers = updated
    return transform, float(struc.rmsd(fitted[inliers], Q[inliers])), int(inliers.sum())


def block_centres(coords, block_lengths):
    out, start = [], 0
    for length in block_lengths:
        out.append(coords[start:start + length].mean(axis=0))
        start += length
    return np.array(out)


def build_peptide_geometry(cif_dir, inhibitors, complexes):
    """Superposed block-centre distances for every inhibitor and its complex's peptide.

    Model independent, so it does not need rebuilding when embeddings change. Every scored
    match is written; the inclusion cut-offs are applied by the analysis script, which also
    derives the family, ligand code and pair counts rather than storing them here.
    """
    peptides = pd.read_parquet(DATA_DIR / "peptide_partners_processed.parquet")
    peptides["pdb"] = peptides["id"].str.split("_").str[0]
    peptides = peptides.merge(complexes, on="pdb", how="left").set_index("family")

    processed = pd.read_parquet(DATA_DIR / "inhibitors_processed.parquet",
                                columns=["id", "X", "block_lengths", "segment_ids"])
    processed["row"] = np.arange(len(processed))
    entry = processed["id"].str.split("_").str[0]
    for column in ("pdb_code", "family", "chain_target"):
        processed[column] = entry.map(inhibitors[column])
    processed = processed[processed["family"].isin(peptides.index)].reset_index(drop=True)
    print(f"{len(processed)} inhibitors across {processed['family'].nunique()} "
          f"protein-peptide complexes")

    transforms, rows = {}, []
    for inhibitor in tqdm(processed.itertuples(), total=len(processed), desc="geometry"):
        peptide = peptides.loc[inhibitor.family]
        inhibitor_path = find_structure(cif_dir, inhibitor.pdb_code)
        peptide_path = find_structure(cif_dir, peptide["pdb"])
        if inhibitor_path is None or peptide_path is None:
            continue
        key = (inhibitor.pdb_code, inhibitor.chain_target,
               peptide["pdb"], peptide["chain_target"])
        try:
            if key not in transforms:
                transforms[key] = superpose_target_chains(
                    str(inhibitor_path), inhibitor.chain_target,
                    str(peptide_path), peptide["chain_target"])
            transform, rmsd, inliers = transforms[key]
        except Exception as error:
            print(f"  superposition failed for {inhibitor.pdb_code}: {error}")
            continue

        segments = np.asarray(inhibitor.segment_ids)
        inhibitor_lengths = np.asarray(inhibitor.block_lengths)
        peptide_lengths = np.asarray(peptide["block_lengths"])
        ligand = (segments == 1) & (inhibitor_lengths > 1)
        partner = peptide_lengths > 1
        if ligand.sum() < 3 or partner.sum() < 3:
            continue

        pocket_atoms = inhibitor_lengths[segments == 0].sum()
        ligand_centres = block_centres(np.stack(inhibitor.X)[pocket_atoms:],
                                       inhibitor_lengths[segments == 1])
        ligand_centres = transform.apply(ligand_centres)[
            inhibitor_lengths[segments == 1] > 1]
        partner_centres = block_centres(np.stack(peptide["X"]), peptide_lengths)[partner]
        geometry = scipy.spatial.distance.cdist(ligand_centres, partner_centres)

        rows.append({
            "inhibitor_row": np.int32(inhibitor.row),
            "peptide_id": f"{peptide['pdb']}_{peptide['chain_partner']}",
            "align_rmsd": np.float32(rmsd),
            "block_coords_dist": geometry.ravel().astype(np.float32),
        })

    out = DATA_DIR / "peptide_inhibitor_geometry.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out.name}: {len(rows)} matches")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cif_dir", required=True,
                        help="directory of 2P2Idb <PDB>.cif or <PDB>.pdb files")
    parser.add_argument("--mesh_dir", default=str(DATA_DIR / "surface_mesh"),
                        help="cache for MSMS meshes and sampled points")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["index", "peptide", "protein", "geometry"])
    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    if not cif_dir.is_dir():
        raise SystemExit(f"--cif_dir is not a directory: {args.cif_dir}")

    metadata = pd.read_csv(DATA_DIR / "metadata.csv")
    complexes = (metadata[["family", "superfamily", "ppi_pdb", "ppi_chain_target",
                           "ppi_chain_partner"]]
                 .drop_duplicates("family")
                 .rename(columns={"ppi_pdb": "pdb", "ppi_chain_target": "chain_target",
                                  "ppi_chain_partner": "chain_partner"}))

    if "index" not in args.skip:
        index = metadata.copy()
        index["pdb_path"] = index["pdb_code"].apply(
            lambda code: str(find_structure(cif_dir, code) or ""))
        index = index[index["pdb_path"] != ""]
        index = index.rename(columns={"chain_target": "chain1", "chain_ligand": "chain2"})
        index.to_csv(DATA_DIR / "inhibitors_index.csv", index=False)
        print(f"wrote inhibitors_index.csv: {len(index)} rows. Now run:\n"
              f"  python -m atomica.data.process_pdbs "
              f"--data_index_file {DATA_DIR / 'inhibitors_index.csv'} "
              f"--out_path {DATA_DIR / 'inhibitors_processed.parquet'} "
              f"--fragmentation_method PS_300")
    if "peptide" not in args.skip:
        build_peptide_partners(cif_dir, complexes)
    if "protein" not in args.skip:
        build_surface_patches(cif_dir, complexes, Path(args.mesh_dir))
    if "geometry" not in args.skip:
        build_peptide_geometry(
            cif_dir, metadata.drop_duplicates("entry_id").set_index("entry_id"),
            complexes)


if __name__ == "__main__":
    main()
