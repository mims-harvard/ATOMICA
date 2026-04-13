"""
extract_chain.py: Extract selected chains from a PDB/CIF file and save to output file.
Adapted from MaSIF implementation by Pablo Gainza - LPDI STI EPFL 2019
Supports both PDB and CIF file formats.
"""
from Bio.PDB import PDBParser, PDBIO, Select, Selection, StructureBuilder
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.SeqUtils import IUPACData
import os

PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


class NotDisordered(Select):
    """Exclude disordered atoms, keeping only the first altloc."""
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


def find_modified_amino_acids(path):
    """
    Find modified amino acids in the PDB (e.g., MSE - selenomethionine).
    Contributed by github user jomimc.
    """
    res_set = set()
    try:
        for line in open(path, 'r'):
            if line[:6] == 'SEQRES':
                for res in line.split()[4:]:
                    res_set.add(res)
        for res in list(res_set):
            if res in PROTEIN_LETTERS:
                res_set.remove(res)
    except:
        # If file is CIF or reading fails, return empty set
        pass
    return res_set


def extract_chain(infilename, outfilename, chain_ids=None):
    """
    Extract specific chains from a PDB or CIF file.

    Args:
        infilename (str): Path to input PDB or CIF file
        outfilename (str): Path to output PDB file
        chain_ids (list or str): Chain ID(s) to extract. If None, extracts all chains.
                                 Can be a single chain ID string or list of chain IDs.

    Returns:
        str: Path to the output file
    """
    # Convert single chain_id to list
    if isinstance(chain_ids, str):
        chain_ids = [chain_ids]

    # Determine file type and use appropriate parser
    file_ext = os.path.splitext(infilename)[1].lower()
    if file_ext == '.cif':
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    # Parse structure
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]

    # Build new structure with selected chains
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load modified amino acids (only for PDB files)
    modified_amino_acids = find_modified_amino_acids(infilename)

    # Extract chains
    # Map long chain IDs to single character for PDB format compatibility
    chain_id_map = {}
    next_chain_letter = ord('A')

    for chain in model:
        if chain_ids is None or chain.get_id() in chain_ids:
            original_chain_id = chain.get_id()

            # For PDB format, chain IDs must be single character
            # If chain ID is longer than 1 character, map it to a single letter
            if len(original_chain_id) > 1:
                new_chain_id = chr(next_chain_letter)
                next_chain_letter += 1
                chain_id_map[original_chain_id] = new_chain_id
            else:
                new_chain_id = original_chain_id
                chain_id_map[original_chain_id] = new_chain_id

            structBuild.init_chain(new_chain_id)
            for residue in chain:
                het = residue.get_id()
                # Include standard residues
                if het[0] == " ":
                    outputStruct[0][new_chain_id].add(residue)
                # Include modified amino acids
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][new_chain_id].add(residue)

    # Save extracted structure
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered())

    return outfilename
