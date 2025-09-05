with open("/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/protein_universe/foldseek-cluster-uniprot-annotation-more-than-3-cluster-30.txt", "r") as f:
    prot_names = f.read().splitlines()
    prot_names = set(prot_names)

input_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_3_2024_11_27.tsv"
output_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_3_2024_11_27_cluster30.fasta"
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        content = line.strip().split('\t')
        id = content[0]
        if id == "Entry":
            continue
        if id not in prot_names:
            continue
        fasta = content[-1]
        modified_line = f">{id}\n{fasta}\n"
        outfile.write(modified_line)

input_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_4_2024_11_27.tsv"
output_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_4_2024_11_27_cluster30.fasta"
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        content = line.strip().split('\t')
        id = content[0]
        if id == "Entry":
            continue
        if id not in prot_names:
            continue
        fasta = content[-1]
        modified_line = f">{id}\n{fasta}\n"
        outfile.write(modified_line)

input_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_5_2024_11_25.tsv"
output_file_path = "/n/holystore01/LABS/mzitnik_lab/Lab/afang/protein_universe/uniprot/uniprotkb_AND_annotation_score_5_2024_11_25_cluster30.fasta"
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        content = line.strip().split('\t')
        id = content[0]
        if id == "Entry":
            continue
        if id not in prot_names:
            continue
        fasta = content[-1]
        modified_line = f">{id}\n{fasta}\n"
        outfile.write(modified_line)