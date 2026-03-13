import requests
import subprocess
import numpy as np
import pandas as pd
import argparse
import pickle
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_uniref_cluster_id(uniprot_id, uniref_level="0.5"):
    url = f"https://rest.uniprot.org/uniref/search?query=uniprot_id:{uniprot_id}%20AND%20identity:{uniref_level}"
    response = requests.get(url, headers={"Accept": "application/json"})
    if response.status_code == 200:
        data = response.json()
        try:
            uniref_cluster_id = data['results'][0]['id']
        except:
            uniref_cluster_id = None
            print(f"UniRef{uniref_level} cluster not found for {uniprot_id}.")
        return uniref_cluster_id
    print(f"UniRef{uniref_level} cluster not found for {uniprot_id}.")
    return None


def fetch_uniref_cluster_members(uniref_cluster_id):
    url = f"https://rest.uniprot.org/uniref/{uniref_cluster_id}"
    response = requests.get(url, headers={"Accept": "application/json"})
    if response.status_code == 200:
        data = response.json()
        ids = []
        if data['memberCount'] > 1:
            for x in data['members']:
                if 'accessions' in x.keys():
                    ids.append(x['accessions'][0])
                elif x['memberIdType'] == 'UniParc':
                    ids.append(x['memberId'])
        else:
            ids = []
        return ids
    else:
        print(f"Failed to fetch UniRef cluster FASTA for {uniref_cluster_id}: {response.status_code}")


def fetch_fasta(id, max_retries=3, timeout=5):
    """Fetch the FASTA for a given UniProt or UniParc ID."""
    if id.startswith("UPI"):
        uniparc = True
        url = f"https://rest.uniprot.org/uniparc/{id}.fasta"
    else:
        uniparc = False
        url = f"https://rest.uniprot.org/uniprotkb/{id}.fasta"
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                text = response.text
                # Fix header if necessary
                if id not in text:
                    if uniparc:
                        text = text.split(' ')
                        text[0] = ">" + id
                        text = " ".join(text)
                    else:
                        text = text.split("|")
                        text[1] = id
                        text = "|".join(text)
                return text  # Return fetched text if successful
            else:
                print(f"Failed to fetch FASTA for {id}: {response.status_code}")
                return None
        except requests.exceptions.Timeout:
            print(f"Timeout occurred for {id}, attempt {attempt + 1}/{max_retries}")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred for {id}: {e}")
        except Exception as e:
            print(f"Error occurred for {id}: {e}")
    print(f"Failed to fetch FASTA for {id} after {max_retries} attempts")
    return None


def uniprot_ids_to_fasta(ids, fasta_file, max_retries=3, timeout=5, max_workers=24):
    """Fetch FASTA sequences concurrently for a list of UniProt/UniParc IDs and save to a file."""
    with open(fasta_file, "w") as f:
        # Use ThreadPoolExecutor to fetch FASTA sequences concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {executor.submit(fetch_fasta, id, max_retries, timeout): id for id in ids}
            for future in as_completed(future_to_id):
                id = future_to_id[future]
                try:
                    fasta_text = future.result()
                    if fasta_text:
                        f.write(fasta_text)  # Write the fetched FASTA text to the file
                except Exception as e:
                    print(f"Error occurred for {id}: {e}")
    return


def clustal_omega_align(fasta_file, output_file):
    cmd = f"clustalo -i {fasta_file} -o {output_file} --force"
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during Clustal Omega alignment: {e}")
    return


def read_fasta(fasta_file):
    seqs = {}
    with open(fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                if "UPI" in line:
                    header = line.strip().split(' ')[0].lstrip(">")
                else:
                    header = line.strip().lstrip(">").split('|')[1]
                seqs[header] = ""
            else:
                seqs[header] += line.strip()
    return seqs


def split_fasta(fasta_in,fasta_out): # removes uniparc sequences from the fasta
    seqs = {}
    with open(fasta_in, "r") as f:
        for line in f:
            if line.startswith(">"):
                if "UPI" in line:
                    uniparc = True
                else:
                    uniparc = False
                    header = line
                    seqs[header] = ""
            else:
                if uniparc == False:
                    seqs[header] += line
    with open(fasta_out, "w") as f:
        for header,seq in seqs.items():
            f.write(header)
            f.write(seq)
    return


def prep_indices_str(indices_str,seqs):
    try:
        df = pd.read_csv(indices_str, sep='\t', header=None, names=['header','indices_str'])
        final_seqs = {}
        final_indices_str = []
        for header,seq in seqs.items():
            df_subset = df[df['header'] == header]
            if header in df_subset['header'].values:
                final_indices_str.append(df_subset['indices_str'].values[0])
                final_seqs[header] = seq
        return final_seqs,final_indices_str
    except FileNotFoundError:
        try:
            print('Index file not found. Parsing as a string...')
            return seqs,[indices_str]*len(seqs)
        except:
            raise ValueError(f"Index file does not exist or is not an acceptable string: {indices_str}")


def index_str_to_arrays(indices_str, seq_len): # convert string of indices to list of indices
    indices = indices_str.split('_')
    seq_len = int(seq_len)+1
    prep_indices = []
    for i in indices:
        if '-' in i:
            if i == '-': # the whole sequence
                return prep_indices+list(range(0,seq_len))
            elif i[0] == '-': # first N residues
                prep_indices += range(0, int(i[1:])+1)
            elif i[-1] == '-': # last N residues
                prep_indices += range(int(i[:-1]), seq_len)
            else: # range of residues
                prep_indices += range(int(i.split('-')[0]), int(i.split('-')[1])+1)
        else:
            prep_indices.append(int(i))
    prep_indices = sorted(list(set(prep_indices)))
    return [x-1 for x in prep_indices]


def alignment_to_pmf(seqs,header,indices_str):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    seqs_array = np.array([list(seq) for seq in seqs.values()])
    template_seq = seqs_array[list(seqs.keys()).index(header)]
    seqs_df = pd.DataFrame(seqs_array[:, np.where(template_seq != "-")[0]])
    indices = index_str_to_arrays(indices_str, seqs_df.shape[1])
    pmf = seqs_df.apply(lambda x: x.value_counts()).fillna(0)
    # add missing columns
    for aa in amino_acids:
        if aa not in pmf.index:
            pmf.loc[aa] = 0
    # check that all indices are present
    final_indices = []
    for i in indices:
        if i in pmf.columns:
            final_indices.append(i)
    pmf = pmf.loc[amino_acids,final_indices]
    pmf = pmf.div(pmf.sum(axis=0), axis=1).to_numpy().T
    return pmf


def save_outputs(outputs,output_dir,names,as_pkl):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if as_pkl != '':
        for i in range(3):
            outf = f'{output_dir}/{as_pkl[i]}.pkl'
            with open(outf, 'wb') as f:
                pickle.dump({name: output for name, output in zip(names[i], outputs[i].values())}, f)
    else:
        for i in range(3):
            [np.save(f'{output_dir}/{name}.npy', output) for name, output in zip(names[i], outputs[i].values())]
    return


def prep_names(headers, num_seqs, indices_str, as_pkl):
    if as_pkl != '':
        as_pkl = [f'MSA.pmf.UniRef{x}.{as_pkl}' for x in ['50','90','100']]
    names50 = [f'{header}.MSA.pmf.UniRef50.num_seqs{num_seqs[i][0]}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
    names90 = [f'{header}.MSA.pmf.UniRef90.num_seqs{num_seqs[i][1]}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
    names100 = [f'{header}.MSA.pmf.UniRef100.num_seqs{num_seqs[i][2]}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
    return [names50,names90,names100],as_pkl


def main(args):
    fasta = args.fasta
    output_dir = args.output_dir
    indices_str = args.indices
    as_pkl = args.as_pkl
    keep_fasta = args.keep_fasta

    seqs = read_fasta(fasta)
    seqs,indices_str = prep_indices_str(indices_str,seqs)
    uniprot_ids = list(seqs.keys())
    if not os.path.exists(f"{output_dir}/fastas"):
        os.makedirs(f"{output_dir}/fastas")
    if not os.path.exists(f"{output_dir}/uniref_ids.txt"):
        uniref_ids = pd.DataFrame(columns=['uniprot_id','uniref50_cluster_id','uniref90_cluster_id','uniref100_cluster_id','uniref50_cluster_members','uniref90_cluster_members','uniref100_cluster_members'])
    else:
        uniref_ids = pd.read_csv(f"{output_dir}/uniref_ids.txt", sep='\t')

    for i,uniprot_id in enumerate(uniprot_ids): # first loop gathers data for all uniprot_ids
        if uniprot_id == 'nan':
            continue
        # Pulling cluster info
        if uniprot_id not in uniref_ids['uniprot_id'].values:
            uniref50_cluster_id = get_uniref_cluster_id(uniprot_id, '0.5')
            uniref90_cluster_id = get_uniref_cluster_id(uniprot_id, '0.9')
            uniref100_cluster_id = get_uniref_cluster_id(uniprot_id, '1.0')
            uniref50_ids = fetch_uniref_cluster_members(uniref50_cluster_id)
            uniref90_ids = fetch_uniref_cluster_members(uniref90_cluster_id)
            uniref100_ids = fetch_uniref_cluster_members(uniref100_cluster_id)
            if uniref50_cluster_id is None or uniref90_cluster_id is None or uniref100_cluster_id is None:
                uniref_ids.loc[len(uniref_ids)] = [uniprot_id, None, None, None, None, None, None]
                uniref_ids.to_csv(f"{output_dir}/uniref_ids.txt", sep='\t', index=False)
                continue
            if uniprot_id not in uniref50_ids:
                uniref50_ids.append(uniprot_id)
            if uniprot_id not in uniref90_ids:
                uniref90_ids.append(uniprot_id)
            if uniprot_id not in uniref100_ids:
                uniref100_ids.append(uniprot_id)
            uniref_ids.loc[len(uniref_ids)] = [uniprot_id, uniref50_cluster_id, uniref90_cluster_id, uniref100_cluster_id, '_'.join(uniref50_ids), '_'.join(uniref90_ids), '_'.join(uniref100_ids)]
            uniref_ids.to_csv(f"{output_dir}/uniref_ids.txt", sep='\t', index=False)
        
        # Pulling FASTAs for cluster
        uniref50_cluster_id = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_id'].values[0]
        if pd.isna(uniref50_cluster_id):
            print(f"UniRef50 cluster not found for {uniprot_id}.")
            continue
        for file in os.listdir(f"{output_dir}/fastas"):
            if file == f"{uniref50_cluster_id}.{uniprot_id}.fasta":
                has_fasta = True
                break
            else:
                has_fasta = False
        if has_fasta != True:
            uniref50_ids = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_members'].values[0].split('_')
            uniref50_cluster_id = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_id'].values[0]
            uniprot_ids_to_fasta(uniref50_ids, f"{output_dir}/fastas/{uniref50_cluster_id}.{uniprot_id}.fasta")
            clustal_omega_align(f"{output_dir}/fastas/{uniref50_cluster_id}.{uniprot_id}.fasta", f"{output_dir}/fastas/{uniref50_cluster_id}.{uniprot_id}.aligned.fasta")

    uniref50_outputs = {}
    uniref90_outputs = {}
    uniref100_outputs = {}
    num_seqs = []
    final_uniprot_ids = []
    final_indices_str = []
    for i,uniprot_id in enumerate(uniprot_ids): # second loop processes the data
        if (uniprot_id == 'nan') | (uniprot_id not in uniref_ids['uniprot_id'].values) | (pd.isna(uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_id'].values[0])):
            continue
        uniref50_cluster_id = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_id'].values[0]
        uniref50_ids = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref50_cluster_members'].values[0].split('_')
        uniref90_ids = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref90_cluster_members'].values[0].split('_')
        uniref100_ids = uniref_ids[uniref_ids['uniprot_id'] == uniprot_id]['uniref100_cluster_members'].values[0].split('_')
        uniref50_seqs = read_fasta(f"{output_dir}/fastas/{uniref50_cluster_id}.{uniprot_id}.aligned.fasta")
        uniref50_outputs[uniprot_id] = alignment_to_pmf(uniref50_seqs, uniprot_id, indices_str[i])
        uniref90_outputs[uniprot_id] = alignment_to_pmf({k:v for k,v in uniref50_seqs.items() if k in uniref90_ids}, uniprot_id, indices_str[i])
        uniref100_outputs[uniprot_id] = alignment_to_pmf({k:v for k,v in uniref50_seqs.items() if k in uniref100_ids}, uniprot_id, indices_str[i])
        if not keep_fasta:
            os.remove(f"{uniref50_cluster_id}.fasta")
            os.remove(f"{uniref50_cluster_id}.aligned.fasta")
        num_seqs.append([len(uniref50_ids),len(uniref90_ids),len(uniref100_ids)])
        final_uniprot_ids.append(uniprot_id)
        final_indices_str.append(indices_str[i])
    names,as_pkl = prep_names(final_uniprot_ids, num_seqs, final_indices_str, as_pkl)
    save_outputs([uniref50_outputs,uniref90_outputs,uniref100_outputs],output_dir,names,as_pkl)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sequences using ESM-2 model.')
    parser.add_argument('--fasta', type=str, help='Input FASTA file')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--indices', type=str, help='Tab delimited text file indicating which positions on each protein to run. Alternatively, a string for the same position in all proteins.',default = '1-')
    parser.add_argument('--as_pkl', type=str, help='Save output as a single pickle file rather than as individual *.npy files. Give output filename prefix to activate string.', default = '')
    parser.add_argument('--keep_fasta', action='store_true', help='Whether to store intermediate UniRef50 FASTA file. Default: False')
    args = parser.parse_args()
    main(args)
