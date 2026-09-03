"""Regenerate the PLIP annotations this tutorial evaluates ATOMICAScore against.

A residue is positive when PLIP reports it in one of five non-covalent interactions with the
ligand. Salt bridges and water bridges are reported by PLIP but are not part of the label.

PLIP is not an ATOMICA dependency and the output is committed, so run this only to relabel or to
label your own structures. Structures not found locally are downloaded from RCSB:

    pip install plip
    python make_plip_labels.py
"""
import argparse
import csv
import os
import sys
import warnings

INTERACTION_TYPES = (
    "Hydrogen Bonds",
    "Hydrophobic Interactions",
    "pi-Stacking",
    "Metal Complexes",
    "Halogen Bonds",
)

AMINO_ACIDS = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP "
                        "TYR VAL".split())

# complex id -> (PDB entry, ligand code, ligand chain, ligand residue number).
# 4yaz is absent on purpose: its receptor is RNA, so it has no amino-acid block to rank.
EXAMPLES = {
    "6llw_A_A_UDP": ("6llw", "UDP", "A", 900),
    "6hrg_A_A_ZN": ("6hrg", "ZN", "A", 301),
}


def pdb_file(entry, pdb_dir, cache):
    """Local PDB file if present, else download the entry into `cache`."""
    local = os.path.join(pdb_dir, f"{entry}.pdb")
    if os.path.exists(local):
        return local
    import urllib.request

    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{entry}.pdb")
    if not os.path.exists(path):
        print(f"downloading {entry} from RCSB")
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{entry}.pdb", path)
    return path


def interactions(site):
    """Yield (interaction type, chain, residue number, residue type) for one binding site."""
    groups = {
        "Hydrogen Bonds": list(site.hbonds_pdon) + list(site.hbonds_ldon),
        "Hydrophobic Interactions": list(site.hydrophobic_contacts),
        "pi-Stacking": list(site.pistacking),
        "Metal Complexes": list(site.metal_complexes),
        "Halogen Bonds": list(site.halogen_bonds),
    }
    for name in INTERACTION_TYPES:
        for item in groups[name]:
            restype = getattr(item, "restype", None)
            if restype is not None:
                yield name, item.reschain, int(item.resnr), restype


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", default=os.path.join(here, "..", "..", "data", "example",
                                                          "example_data"))
    parser.add_argument("--out", default=os.path.join(here, "..", "..", "data", "example",
                                                      "example_plip_labels.csv"))
    parser.add_argument("--cache", default=os.path.join(here, "pdb_cache"),
                        help="where to put entries downloaded because they are not in --pdb_dir")
    args = parser.parse_args()

    try:
        from plip.structure.preparation import PDBComplex
    except ImportError:
        sys.exit("PLIP is not installed. Run `pip install plip`, or use the committed CSV.")

    rows = []
    for complex_id, (entry, lig_code, lig_chain, lig_resnum) in EXAMPLES.items():
        structure = PDBComplex()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure.load_pdb(pdb_file(entry, args.pdb_dir, args.cache))
            structure.analyze()
        # A deposited structure often holds several copies of a ligand; take the one used here.
        site = f"{lig_code}:{lig_chain}:{lig_resnum}"
        if site not in structure.interaction_sets:
            sys.exit(f"{complex_id}: no site {site}. Found {sorted(structure.interaction_sets)}")
        seen = set()
        for kind, chain, resnum, restype in interactions(structure.interaction_sets[site]):
            if restype not in AMINO_ACIDS or (kind, chain, resnum) in seen:
                continue
            seen.add((kind, chain, resnum))
            rows.append({"complex_id": complex_id, "chain": chain, "residue": resnum,
                         "restype": restype, "interaction_type": kind,
                         "pdb_residue": f"{chain}_{resnum}"})
        print(f"{complex_id}: {len(seen)} annotated residues at {site}")

    rows.sort(key=lambda r: (r["complex_id"], r["chain"], r["residue"], r["interaction_type"]))
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["complex_id", "chain", "residue", "restype",
                                                    "interaction_type", "pdb_residue"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
