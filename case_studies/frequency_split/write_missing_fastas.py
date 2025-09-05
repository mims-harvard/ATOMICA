
# for any missing fastas rewrite them
from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio.SeqIO.FastaIO import FastaWriter
import os
from tqdm import tqdm
import pickle

def pdb_to_fasta(pdb_filename):
    structure_id = os.path.basename(pdb_filename).split('.')[0]

    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)

    # Parse the structure
    structure = parser.get_structure("protein_structure", pdb_filename)


    output = {}

    # Open the FASTA file for writing
    for model in structure:
        for chain in model:
            sequence = ""
            for residue in chain:
                # Ensure the residue is an amino acid
                if PDB.is_aa(residue, standard=True):
                    sequence += seq1(residue.get_resname())
            
            # Create a sequence record for the chain
            seq_record = SeqRecord(
                Seq(sequence),
                id=f"{structure_id}_{chain.id}",
                description=f"Chain {chain.id}"
            )

            output[seq_record.id] = str(seq_record.seq)
    return output

def write_missing_fastas(dataset_path, fasta_path, rec_path):
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    with open(fasta_path, "r") as f:
        lines = f.readlines()
        pion_fastas = {k.strip().replace(">", ""):v.strip().replace(">", "") for k,v in zip(lines[::2], lines[1::2])}

    missing_fastas = {}
    for item in tqdm(dataset, total=len(dataset)):
        chains = set(pdb_index.split('_')[0] for pdb_index in item['block_to_pdb_indexes'].values())
        pdb_file = "_".join(item['id'].split("_")[:2]).replace(".pdb", "")
        for chain in chains:
            if f"{pdb_file}_{chain}" not in pion_fastas and f"{pdb_file}_{chain}" not in missing_fastas:
                new_fastas = pdb_to_fasta(f"{rec_path}/{pdb_file}.pdb")
                missing_fastas.update(new_fastas)

    all_fastas = {**pion_fastas, **missing_fastas}
    with open(fasta_path, "w") as f:
        for k,v in all_fastas.items():
            f.write(f">{k}\n{v}\n")
        
    print(fasta_path, "added", len(missing_fastas))

if __name__ == "__main__":
    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/Pion_nonredund.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/Pion/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/Pion/nonredund_rec",
    )

    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/PL_nonredund.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PL/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PL/nonredund_rec",
    )

    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/PRNA_nonredund.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PRNA/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PRNA/nonredund_rec",
    )

    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/PDNA_nonredund.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PDNA/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PDNA/nonredund_rec",
    )

    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/Ppeptide_nonredund.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PIII/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PIII/nonredund_rec",
    )

    write_missing_fastas(
        dataset_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/QBioLiP/QBioLiP-06-2024/PP_nonredund_fixed_ids.pkl",
        fasta_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PPI/nonredund.fasta",
        rec_path = "/n/netscratch/mzitnik_lab/Lab/afang/InteractNN/raw_QBioLiP_06_2024/PPI/nonredund_pdb",
    )


