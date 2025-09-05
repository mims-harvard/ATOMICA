import os
from openai import AzureOpenAI
import re
import random
import pandas as pd
from tqdm import tqdm
random.seed(42)

API_KEY = "f65f6d81f08947a488e4599b08110058"
API_ENDPOINT = "https://azure-ai-dev.hms.edu"
API_VERSION = "2024-10-01-preview"

client = AzureOpenAI(
  azure_endpoint=API_ENDPOINT, 
  api_key=API_KEY,
  api_version=API_VERSION,
)

def get_question(row_query, row_random):
    question = f"""
    You are an expert in biology. Please help me with the following question by thinking step by step:

    You will be given amino acid protien sequence in the format of one letter amino acid codes if there are multiple chains in the protein they are separated by the "|" character. 
    The nucleic acid binding sites of the protein in the format of <pdb_position>_<three_letter_amino_acid_code>.
    
    You will be given this information for a query protein sequence and two sets of proteins.
    Step 1: Compare the nucleic acid binding domains in the given amino acid sequence with the nucleic acid binding domains in the protein sequences of set 1 and set 2.
    Step 2: Based on the comparison, determine if the nucleic acid binding domains in the given amino acid sequence are more similar to the nucleic acid binding domains in set 1 or set 2.
    Step 3: Provide your answer in the format: Final answer: Set 1 or Set 2.

    Query protein sequence: {row_query['query_fasta']}
    Query protein binding site: {row_query['query_binding_site']}
    """

    retreived_proteins = []
    for i in range(10):
        retreived_proteins.append(f"Protein {i+1} sequence: {row_query[f'{i}_fasta']}")
        retreived_proteins.append(f"Protein {i+1} binding site: {row_query[f'{i}_binding_site']}")
    retreived_proteins = "\n".join(retreived_proteins)

    random_retreived_proteins = []
    for i in range(10):
        random_retreived_proteins.append(f"Protein {i+1} sequence: {row_random[f'{i}_fasta']}")
        random_retreived_proteins.append(f"Protein {i+1} binding site: {row_random[f'{i}_binding_site']}")
    random_retreived_proteins = "\n".join(random_retreived_proteins)
    
    retreived_order = [1, 2]
    random.shuffle(retreived_order)
    if retreived_order[0] == 1:
        question += f"""
        Set 1:
        {retreived_proteins}

        Set 2:
        {random_retreived_proteins}
        """
        correct_answer = 1
    else:
        question += f"""
        Set 1:
        {random_retreived_proteins}

        Set 2:
        {retreived_proteins}
        """
        correct_answer = 2
    
    return question, correct_answer
    

def main(query_df):
    results = []
    for index, row in tqdm(query_df.iterrows(), total=query_df.shape[0]):
        row_query = row
        row_random = query_df.sample(n=1).iloc[0]
        question, correct_answer = get_question(row_query, row_random)

        message = [{"role": "system", "content": "You are a helpful AI assistant."}, 
                {"role": "user", "content": question}]
        
        print(question)

        response = client.chat.completions.create(
            model='gpt-4o', # Model deployment name
            messages = message,        
            temperature=0.5,
            max_tokens=4096,
        )

        content = response.choices[0].message.content

        print(content)

        match = re.search(r'Final answer:\s*Set (\d)', content)
        if match:
            final_answer = int(match.group(1))
            results.append(final_answer==correct_answer)
            print(f"Correct answer: {correct_answer}, Final answer: {final_answer}")
        else:
            results.append(None)
            print("Final answer not found in content.")
    
    query_df['correct'] = results
    return query_df

if __name__ == "__main__":
    query_file = "/n/holylabs/LABS/mzitnik_lab/Users/afang/GET/case_studies/protein_universe/retrieved_embeddings/foldseek_cluster_is_dark_90_plddt_PeSTo_80_nucleic_acid.tsv"
    query_df = pd.read_csv(query_file, sep="\t")
    results = main(query_df)
    results.to_csv(query_file.replace(".tsv", "_results.tsv"), sep="\t", index=False)