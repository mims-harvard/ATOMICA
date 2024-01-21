from ccdc import search
from ccdc.molecule import Molecule
from ccdc.io import CrystalReader
from ccdc.utilities import _private_importer

with _private_importer() as pi:
    pi.import_ccdc_module("UtilitiesLib")
    pi.import_ccdc_module("MathsLib")
    pi.import_ccdc_module("ChemistryLib")
    pi.import_ccdc_module("ChemicalAnalysisLib")
    pi.import_ccdc_module("FileFormatsLib")
    pi.import_ccdc_module("MotifSearchLib")
    pi.import_ccdc_module("PackingSimilarityLib")

from glob import glob
import os
import multiprocessing
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
import argparse
import sys
import pickle

PROJ_DIR = os.path.join(
    os.path.split(os.path.abspath(__file__))[0],
    '..', '..'
)
print(f'Project directory: {PROJ_DIR}')
sys.path.append(PROJ_DIR)

from data.dataset import Block, Atom, VOCAB
from data.dataset import blocks_interface, blocks_to_data
from utils.logger import print_log

INTERMOLECULAR_CUTOFF = 4
CHUNK_SIZE = 10000


def process_crystals(
    csd_data_directory,
    processed_dir,
    num_workers=16,
    search_settings_param=None,
    intermolecular_cutoff=INTERMOLECULAR_CUTOFF,
    chunk_size=CHUNK_SIZE,
):
    """
    Process all crystals in the CSD data directory to create discrete molecular shells and save the results to a file.
    """
    # Create save directory
    os.makedirs(processed_dir, exist_ok=False)

    # set search settings
    if search_settings_param is None:
        search_settings_param = {
            "only_organic": True,
            "not_polymeric": True,
            "has_3d_coordinates": True,
            "no_disorder": True,
            "no_errors": True,
            "no_metals": True,
        }
    print_log(f"No search settings provided, using default settings: {search_settings_param}", level="INFO")

    # Find sqlite files in the CSD data directory
    csd_data_dirs = glob(csd_data_directory + "/*.sqlite")
    csd_data_dirs.sort(key=lambda x: x.lower())
    csd_data_dirs.reverse()

    for csd_data_dir in csd_data_dirs:
        if num_workers > 1:
            print_log(f"Processing {csd_data_dir}", level="INFO")
            crystal_reader_len = len(CrystalReader(csd_data_dir))
            with multiprocessing.Pool(num_workers) as pool:
                params = [
                    (
                        idx,
                        idx + crystal_reader_len // num_workers + 1,
                        search_settings_param,
                        csd_data_dir,
                        processed_dir,
                        intermolecular_cutoff,
                        chunk_size,
                    )
                    for idx in range(
                        0, crystal_reader_len, crystal_reader_len // num_workers + 1
                    )
                ]
                list(
                    tqdm(
                        pool.imap_unordered(get_crystal_features_pool, params),
                        total=len(params),
                    )
                )
        else:
            print_log(f"Processing {csd_data_dir}", level="INFO")
            crystal_reader = CrystalReader(csd_data_dir)
            settings = get_search_settings(search_settings_param)
            results = {}
            for idx in tqdm(range(len(crystal_reader))):
                crystal = crystal_reader[idx]
                get_crystal_features(
                    idx,
                    crystal,
                    settings,
                    results,
                    csd_data_dir,
                    processed_dir,
                    intermolecular_cutoff,
                    chunk_size,
                )
            # Save any remaining results
            if len(results) > 0:
                save_results(
                    results,
                    csd_data_dir,
                    processed_dir,
                    chunk_size,
                    len(crystal_reader),
                )


def get_search_settings(search_settings_param):
    settings = search.Search.Settings()
    settings.only_organic = search_settings_param["only_organic"]
    settings.not_polymeric = search_settings_param["not_polymeric"]
    settings.has_3d_coordinates = search_settings_param["has_3d_coordinates"]
    settings.no_disorder = search_settings_param["no_disorder"]
    settings.no_errors = search_settings_param["no_errors"]
    settings.no_metals = search_settings_param["no_metals"]
    return settings


def get_crystal_features_pool(params):
    (
        start_idx,
        end_idx,
        search_settings_param,
        csd_data_dir,
        processed_dir,
        intermolecular_cutoff,
        chunk_size,
    ) = params
    settings = get_search_settings(search_settings_param)
    results = {}
    crystal_reader = CrystalReader(csd_data_dir)
    end_idx = min(end_idx, len(crystal_reader))
    for idx in range(start_idx, end_idx):
        crystal = crystal_reader[idx]
        get_crystal_features(
            idx,
            crystal,
            settings,
            results,
            csd_data_dir,
            processed_dir,
            intermolecular_cutoff,
            chunk_size,
        )
    if len(results) > 0:
        save_results(results, csd_data_dir, processed_dir, chunk_size, end_idx)


def get_crystal_features(
    idx,
    crystal,
    settings,
    results,
    csd_data_dir,
    processed_dir,
    intermolecular_cutoff,
    chunk_size,
):
    if settings.test(crystal):
        results.update(process_crystal_entry(crystal, intermolecular_cutoff))
    if len(results) >= chunk_size:
        print_log(f"Saving results at crystal reader index {idx}", level="INFO")
        save_results(results, csd_data_dir, processed_dir, chunk_size, idx)


def save_results(results, csd_data_dir, processed_dir, chunk_size, save_index):
    csd_data_dir = csd_data_dir.split("/")[-1].strip(".sqlite")
    fname = f"{processed_dir}/{csd_data_dir}_idx_{save_index}_chunk_{chunk_size}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(list(results.values()), f)
    print_log(f"Saved file at {fname}", level="INFO")
    results.clear()


def blocks_from_molecule(molecule):
    """
    Each atom is its own block
    """
    blocks = []
    for atom in molecule.heavy_atoms:
        coords = [atom.coordinates.x, atom.coordinates.y, atom.coordinates.z]
        element = atom.atomic_symbol
        atom_name = element
        # input(element)
        atom = Atom(
            atom_name=atom_name,
            coordinate=coords,
            element=element,
            pos_code=VOCAB.atom_pos_sm
        )
        blocks.append(Block(
            symbol=atom.element.lower(),
            units=[atom]
        ))
    return blocks


def process_one(data_idx, molecule1, molecule2, interface_dist_th):
    item = {}
    item['id'] = data_idx  # <csd identifier>_<smiles1>_<conformer index>_<smiles2>_<molecule index>
    item['affinity'] = { 'neglog_aff': -1.0 }

    blocks1 = blocks_from_molecule(molecule1)
    blocks2 = blocks_from_molecule(molecule2)

    # construct pockets
    interface_blocks, _ = blocks_interface(blocks1, blocks2, interface_dist_th)
    if len(interface_blocks) < 5:
        print_log(f'ERROR: {data_idx} has an insufficient interface', level="DEBUG")
        return None
    
    data = blocks_to_data(blocks1, blocks2)
    for key in data:
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    item['data'] = data
    print_log(f'{data_idx} was successful', level="DEBUG")
    return item


def process_crystal_entry(crystal, intermolecular_cutoff):
    """Builds data entries of molecules interacting in crystal

    Args:
        crystal (ccdc.crystal.Crystal): Crystal structure to build the molecular shell around
    """
    crystal_view = ChemistryLib.CrystalStructureView.instantiate(crystal._crystal)
    distance_type = "actual"
    distance_range = (0, intermolecular_cutoff)
    contact_criterion = ChemistryLib.CombinedCriterion()
    contact_criterion.set_only_strongest(True)
    contact_criterion.set_min_tolerance(min(distance_range))
    contact_criterion.set_max_tolerance(max(distance_range))
    if distance_type.lower() == "vdw":
        contact_criterion.set_vdw_corrected(True)
    else:
        contact_criterion.set_vdw_corrected(False)

    data = {}
    for idx in range(
        crystal_view.nmolecules()
    ):  # Can have different conformers in the on molecule
        molecule_subset = crystal_view.molecule(idx)

        # Find the molecular contacts
        molecular_contacts = crystal_view.find_contacts(
            molecule_subset, contact_criterion
        )
        if len(molecular_contacts) == 0:
            print_log(f"No molecular contacts found for {crystal.identifier}", level="DEBUG")
            continue

        central_molecule = Molecule(_molecule=molecular_contacts[0].molecule1())
        if not check_valid_molecule(central_molecule):
            continue
        
        for mol_idx, molecular_contact in enumerate(molecular_contacts):
            periphery_molecule = Molecule(_molecule=molecular_contact.molecule2())
            if not check_valid_molecule(periphery_molecule):
                continue
            data_idx = f"{crystal.identifier}_{central_molecule.smiles}_{idx}_{periphery_molecule.smiles}_{mol_idx}"
            entry = process_one(data_idx, central_molecule, periphery_molecule, intermolecular_cutoff)
            if entry is not None:
                data[data_idx] = entry
    return data

def check_valid_molecule(molecule):
    num_heavy_atoms = len(molecule.heavy_atoms)
    if num_heavy_atoms < 6 or num_heavy_atoms > 50:
        return False
    return True


def get_crystal_from_identifier(csd_data_directory, identifier):
    csd_data_dirs = glob(csd_data_directory + "/*.sqlite")
    csd_data_dirs.sort(key=lambda x: x.lower())

    crystal_entry = None
    for csd_data_dir in csd_data_dirs:
        crystal_reader = CrystalReader(csd_data_dir)
        try:
            crystal_entry = crystal_reader.crystal(identifier)
        except RuntimeError as error:
            if "CSDSQLDatabase::entry( DatabaseEntryIdentifier )():" in str(error):
                continue
            else:
                raise error
    if crystal_entry is None:
        raise RuntimeError(f"Could not find crystal with identifier {identifier}")
    return crystal_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csd_data_directory', type=str)
    parser.add_argument('--processed_dir', type=str, default='./datasets/CSD')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    process_crystals(
        csd_data_directory=args.csd_data_directory,
        processed_dir=args.processed_dir,
        num_workers=args.num_workers,
    )
