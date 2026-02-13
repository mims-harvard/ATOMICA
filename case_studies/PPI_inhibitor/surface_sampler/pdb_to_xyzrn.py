"""
pdb_to_xyzrn.py: Convert PDB file to XYZRN format for MSMS input
Adapted from MaSIF implementation by Pablo Gainza - LPDI STI EPFL 2019
XYZRN format: x y z radius density atom_id
"""
from Bio.PDB import PDBParser

# Van der Waals radii for atoms (from MaSIF chemistry.py)
RADII = {
    "N": "1.540000",
    "O": "1.400000",
    "C": "1.740000",
    "H": "1.200000",
    "S": "1.800000",
    "P": "1.800000",
    "Z": "1.39",
    "X": "0.770000",  # Radii of CB or CA in disembodied case
}

# Polar hydrogens for each amino acid (from MaSIF chemistry.py)
POLAR_HYDROGENS = {
    "ALA": ["H"],
    "GLY": ["H"],
    "SER": ["H", "HG"],
    "THR": ["H", "HG1"],
    "LEU": ["H"],
    "ILE": ["H"],
    "VAL": ["H"],
    "ASN": ["H", "HD21", "HD22"],
    "GLN": ["H", "HE21", "HE22"],
    "ARG": ["H", "HH11", "HH12", "HH21", "HH22", "HE"],
    "HIS": ["H", "HD1", "HE2"],
    "TRP": ["H", "HE1"],
    "PHE": ["H"],
    "TYR": ["H", "HH"],
    "GLU": ["H"],
    "ASP": ["H"],
    "LYS": ["H", "HZ1", "HZ2", "HZ3"],
    "PRO": [],
    "CYS": ["H"],
    "MET": ["H"],
}


def pdb_to_xyzrn(pdb_filename, xyzrn_filename):
    """
    Convert PDB file to XYZRN format for MSMS.

    Args:
        pdb_filename (str): Path to input PDB file
        xyzrn_filename (str): Path to output XYZRN file

    Returns:
        str: Path to the XYZRN file
    """
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(pdb_filename, pdb_filename)

    with open(xyzrn_filename, "w") as outfile:
        for atom in struct.get_atoms():
            name = atom.get_name()
            residue = atom.get_parent()

            # Ignore HETATM residues
            if residue.get_id()[0] != " ":
                continue

            resname = residue.get_resname()
            reskey = residue.get_id()[1]
            chain = residue.get_parent().get_id()
            atomtype = name[0]

            # Determine atom color for visualization
            color = "Green"
            coords = None

            # Only process atoms with known radii and residues with polar hydrogen info
            if atomtype in RADII and resname in POLAR_HYDROGENS:
                if atomtype == "O":
                    color = "Red"
                if atomtype == "N":
                    color = "Blue"
                if atomtype == "H":
                    if name in POLAR_HYDROGENS[resname]:
                        color = "Blue"  # Polar hydrogens

                coords = "{:.06f} {:.06f} {:.06f}".format(
                    atom.get_coord()[0], atom.get_coord()[1], atom.get_coord()[2]
                )

                insertion = "x"
                if residue.get_id()[2] != " ":
                    insertion = residue.get_id()[2]

                full_id = "{}_{:d}_{}_{}_{}_{}".format(
                    chain, residue.get_id()[1], insertion, resname, name, color
                )

                # Write XYZRN line: x y z radius density atom_id
                outfile.write(coords + " " + RADII[atomtype] + " 1 " + full_id + "\n")

    return xyzrn_filename
