import pandas as pd
from pathlib import Path
import os

DATA_DIR = Path(__file__).parent

# Global variables to store loaded data
_smiles_dict = None
_inchikey_dict = None

def _load_smiles_data():
    """Load SMILES data if not already loaded."""
    global _smiles_dict
    if _smiles_dict is None:
        try:
            smiles_df = pd.read_csv(
                os.path.join(DATA_DIR, "pdb_chemical_components_smiles.txt"), 
                sep="\t", 
                names=["smiles", "ccd_ligand_code", "name"]
            )
            _smiles_dict = dict(zip(smiles_df["ccd_ligand_code"], smiles_df["smiles"]))
        except FileNotFoundError as e:
            print(f"Warning: Could not load SMILES data: {e}")
            _smiles_dict = {}
    return _smiles_dict

def _load_inchikey_data():
    """Load InChIKey data if not already loaded."""
    global _inchikey_dict
    if _inchikey_dict is None:
        try:
            inchikey_df = pd.read_csv(
                os.path.join(DATA_DIR, "pdb_chemical_components_inchikey.txt"), 
                sep="\t", 
                names=["inchikey", "ccd_ligand_code", "name"]
            )
            _inchikey_dict = dict(zip(inchikey_df["ccd_ligand_code"], inchikey_df["inchikey"]))
        except FileNotFoundError as e:
            print(f"Warning: Could not load InChIKey data: {e}")
            _inchikey_dict = {}
    return _inchikey_dict

def get_ligand_smiles(ligand_code: str) -> str:
    """
    Get the SMILES string for a given ligand code.
    """
    smiles_dict = _load_smiles_data()
    
    if not smiles_dict:
        raise FileNotFoundError("SMILES data not loaded")
    
    if ligand_code not in smiles_dict:
        raise ValueError(f"Ligand code {ligand_code} not found in SMILES data")
    
    return smiles_dict[ligand_code]

def get_ligand_inchikey(ligand_code: str) -> str:
    """
    Get the InChIKey for a given ligand code.
    """
    inchikey_dict = _load_inchikey_data()
    
    if not inchikey_dict:
        raise FileNotFoundError("InChIKey data not loaded")
    
    if ligand_code not in inchikey_dict:
        raise ValueError(f"Ligand code {ligand_code} not found in InChIKey data")
    
    return inchikey_dict[ligand_code]