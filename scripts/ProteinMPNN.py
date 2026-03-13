import pandas as pd
import torch
import numpy as np
import subprocess
import os
import argparse
import pickle
import concurrent.futures
import time
from Bio.PDB import PDBParser

### THE MPNN RUN IS WORKING, THE DATA ORGANIZATION IS NOT


def read_index_file(indices_str):
    df = pd.read_csv(indices_str, sep='\t', header=None, names=['header','indices_str'])
    combined_indices = []
    for header in df['header'].unique():
        subset = df[df['header']==header]
        all_indices = '_'.join(subset['indices_str'])
        all_indices = list(set([int(x) for x in all_indices.split('_')]))
        all_indices = '_'.join([str(x) for x in sorted(all_indices)])
        combined_indices.append(pd.DataFrame({'header':[header],'indices_str':[all_indices]}))
    return pd.concat(combined_indices)

def prep_indices_str(inputs,indices_str):
    try:
        df = read_index_file(indices_str)
        final_indices_str = []
        headers = []
        final_inputs = []
        for df_index,df_item in enumerate(df['header']):
            for pointer in inputs:
                if df_item in pointer:
                    headers.append(df_item)
                    final_inputs.append(pointer)
                    final_indices_str.append(df['indices_str'].iloc[df_index])
        return headers,final_inputs,final_indices_str
    except FileNotFoundError:
        try:
            print('Index file not found. Parsing as a string...')
            #headers = [x.split('/')[-1].split('-')[1] for x in inputs] ### FOR ALPHAFOLD
            headers = [x.split('/')[-1].split('.pdb')[0] for x in inputs] ### FOR GENERAL
            return headers,inputs,[indices_str]*len(inputs)
        except:
            raise ValueError(f"Index file does not exist or is not an acceptable string: {indices_str}")

def prep_names(headers, operations, model_choice, mode, indices_str,as_pkl): # generate names for output files
    operations = operations.split('_')
    all_names = {}
    if as_pkl != '':
        as_pkl = {operation: f'{model_choice}.{mode}.{operation}.{as_pkl}' for operation in operations}
    for operation in operations:
        if operation == "x":
            names = [f'{header}.{model_choice}.{mode}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
        else:
            func = getattr(np, operation, None)
            if callable(func):
                names = [f'{header}.{model_choice}.{mode}.{operation}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
            else:
                print(f'Invalid numpy operation: {operation}. Skipping...')
        all_names[operation] = names
    return all_names,as_pkl # {operation: [name1, name2, ...]}, "" or {operation: filename}

def index_str_to_arrays(indices_str, seq_len): # convert string of indices to list of indices
    indices = indices_str.split('_')
    seq_len = int(seq_len)+1
    prep_indices = []
    is_trimmed = False
    for i in indices:
        if '-' in i:
            if i == '-': # the whole sequence
                return prep_indices+list(range(0,seq_len))
            elif i[0] == '-': # first N residues
                prep_indices += range(0, int(i[1:])+1)
            elif i[-1] == '-': # last N residues
                prep_indices += range(int(i[:-1]), seq_len)
            else: # range of residues
                s = int(i.split('-')[0])
                e = int(i.split('-')[1])+1
                if e > seq_len:
                    is_trimmed = True
                    e = seq_len
                prep_indices += range(s,e)
        else:
            if int(i) > seq_len:
                is_trimmed = True
            else:
                prep_indices.append(int(i))
    if is_trimmed:
        print("Positions were requested that exceed the length of the input, trimming to fit the input...")
    prep_indices = sorted(list(set(prep_indices)))
    return prep_indices

def run_mpnn_logits(proteinmpnn_dir, pdb_file, type_mpnn, model_checkpoint, device, indices_str, use_sequence, single_aa_score, output_dir):
    parser = PDBParser()
    structure = parser.get_structure('your_structure', pdb_file)
    seq_len = max(int(residue.id[1]) for residue in structure.get_residues()) - min(int(residue.id[1]) for residue in structure.get_residues())
    indices = index_str_to_arrays(indices_str,seq_len)
    redesigned_residues = (' ').join([f'A{str(x)}' for x in indices])
    indices = [x-1 for x in indices] # 0 indexing
    cwd = os.getcwd() # Save current working directory
    os.chdir(proteinmpnn_dir) # Change directory to where MPNN scripts are centered at
    command = (
        'python ./score.py '
        f'--model_type {type_mpnn} '
        f'--checkpoint_{type_mpnn} {os.path.join("model_params", model_checkpoint)} '
        '--seed 42 '
        #f"--single_aa_score {int(single_aa_score)} "
        f'--out_folder {os.path.join(output_dir, "ProteinMPNN")} '
        f'--pdb_path {pdb_file} '
        f"--use_sequence {int(use_sequence)} "
        '--batch_size 5 '
        '--number_of_batches 1 ' # This might not even be necessary... could check this using FireProtDB
        f'--device {device} '
        f"--redesigned_residues '{redesigned_residues}' "
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate() # for debugging
    os.chdir(cwd) # Change directory back to original
    try:
        output = torch.load(f'{os.path.join(output_dir, "ProteinMPNN", ".".join(pdb_file.split("/")[-1].split(".")[:-1]))}.pt')
        logits = output['logits'].mean(axis=0)[indices]
        return logits
    except:
        print(f'Error in {pdb_file}')
        return None

def run_mpnn_embeds(proteinmpnn_dir, pdb_file, type_mpnn, model_checkpoint, device, indices_str, use_sequence, single_aa_score, output_dir):
    parser = PDBParser()
    structure = parser.get_structure('your_structure', pdb_file)
    seq_len = max(int(residue.id[1]) for residue in structure.get_residues()) - min(int(residue.id[1]) for residue in structure.get_residues())
    indices = index_str_to_arrays(indices_str,seq_len)
    redesigned_residues = (' ').join([f'A{str(x)}' for x in indices])
    indices = [x-1 for x in indices] # 0 indexing
    cwd = os.getcwd() # Save current working directory
    os.chdir(proteinmpnn_dir) # Change directory to where MPNN scripts are centered at
    command = (
        'python ./run_embeds.py '
        f'--model_type {type_mpnn} '
        f'--checkpoint_{type_mpnn} {os.path.join("model_params", model_checkpoint)} '
        '--seed 42 '
        f'--out_folder {os.path.join(output_dir, "ProteinMPNN")} '
        f'--pdb_path {pdb_file} '
        f'--device {device} '
        #f"--redesigned_residues '{redesigned_residues}' " # potentially reinclude
        f'--get_backbone_embeddings 1 '
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    out, err = process.communicate() # for debugging
    os.chdir(cwd) # Change directory back to original
    try:
        # fix string
        output = np.load(f'{output_dir}/ProteinMPNN/backbone_embeddings/{".".join(pdb_file.split("/")[-1].split(".")[:-1])}.npy')
        #
        return output[indices]
    except:
        print(f'Error in {pdb_file}')
        return None

def perform_operations(outputs,mode,operations):
    if mode == 'logits':
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        mpnn_alphabet = {"A": 0,"C": 1,"D": 2,"E": 3,"F": 4,"G": 5,"H": 6,"I": 7,"K": 8,"L": 9,
        "M": 10,"N": 11,"P": 12,"Q": 13,"R": 14,"S": 15,"T": 16,"V": 17,"W": 18,"Y": 19
        }
        new_outputs = []
        for aa in amino_acids:
            new_outputs.append(outputs[:,mpnn_alphabet[aa]])
        new_outputs = [np.array(new_outputs).T]

    elif mode == 'embeds': # replace with multiple possible operations later
        new_outputs = outputs
    operations = operations.split('_')
    operations_outputs = {key: None for key in operations}
    for operation in operations:
        if operation != 'x':
            func = getattr(np, operation, None)
            if callable(func):
                operations_outputs[operation] = [func(x, axis=0) for x in new_outputs]
        else:
            operations_outputs[operation] = new_outputs
    return operations_outputs # {operation: [output1, output2, ...]}


def save_outputs(outputs,output_dir,names,as_pkl): # save outputs as individual files or as a single pickle
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for operation in names:
        if as_pkl != '':
            outf = os.path.join(output_dir,f'{as_pkl[operation]}.pkl')
            with open(outf, 'wb') as f:
                pickle.dump({name: output for name, output in zip(names[operation], outputs[operation])}, f)
        else:
            [np.save(f'{output_dir}/{name}.npy', output) for name, output in zip(names[operation], outputs[operation])]
    return

def main(args):
    proteinmpnn_dir = args.proteinmpnn_dir
    input_pointer = args.input_pointer
    output_dir = args.output_dir
    device = args.device
    type_mpnn = args.type_mpnn
    model_checkpoint = args.model_checkpoint
    indices_str = args.indices_str
    use_sequence = args.use_sequence
    as_pkl = args.as_pkl
    single_aa_score = args.single_aa_score
    num_workers = args.num_workers
    mode = args.mode
    operations = args.operations
    chkpt_num = args.chkpt_num

    if os.path.isfile(input_pointer):
        inputs = [input_pointer]
    elif os.path.isdir(input_pointer):
        inputs = [os.path.join(input_pointer, f) for f in os.listdir(input_pointer)]
    headers,inputs,indices_str = prep_indices_str(inputs,indices_str)

    model_checkpoint_name = model_checkpoint.split('/')[-1].split(".pt")[0]



    ### ADD CHECKPOINTING LOGIC HERE
    for i in range(chkpt_num):
        if chkpt_num == 1: # if only one checkpoint...
            s = 0
            e = len(headers)
            chkpt_str = ''
        elif i == chkpt_num-1: # if the last checkpoint...
            s = i*len(headers)//chkpt_num
            e = len(headers)
            chkpt_str = f'.chkpt_{chkpt_num}'
        else:
            s = i*len(headers)//chkpt_num
            e = (i+1)*len(headers)//chkpt_num
            chkpt_str = f'.chkpt_{i+1}'
        headers_chunk = headers[s:e]
        inputs_chunk = inputs[s:e]
        indices_str_chunk = indices_str[s:e]
        names,as_pkl_dict = prep_names(headers_chunk, operations, f'{type_mpnn}.{model_checkpoint_name}', mode, indices_str_chunk, as_pkl+chkpt_str)


        # make directory {output_dir}/ProteinMPNN/ if it does not exist
        os.makedirs(f'{output_dir}/ProteinMPNN/', exist_ok=True)

        outputs = {x: [] for x in operations.split('_')}
        names_rearranged = {x: [] for x in operations.split('_')}
        args_list = [(proteinmpnn_dir, pdb, type_mpnn, model_checkpoint, device, indices_str_chunk[i], use_sequence, single_aa_score, output_dir) for i,pdb in enumerate(inputs_chunk)]
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            if mode == 'embeds':
                func = run_mpnn_embeds
            elif mode == 'logits':
                func = run_mpnn_logits

            future_to_name = {executor.submit(func, *args): i for args, i in zip(args_list, range(len(args_list)))}
            for future in concurrent.futures.as_completed(future_to_name):
                i = future_to_name[future]
                result = future.result()
                if result is not None:
                    future_outputs = perform_operations(result, mode, operations)
                    for operation in future_outputs:
                        outputs[operation] += future_outputs[operation]
                        names_rearranged[operation] += [names[operation][i]]

        save_outputs(outputs,output_dir,names_rearranged,as_pkl_dict)
        print(f'{num_workers} worker: {time.time()-start}')

    ###

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverse fold using MPNN.')
    parser.add_argument('--proteinmpnn_dir', type=str, help='Path to the proteinmpnn directory')
    parser.add_argument('--input_pointer', type=str, help='Input pointer to the pdb file or directory containing pdb files')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--device', type=str, help='Specify the Cuda device index', default = '0')
    parser.add_argument('--mode', type=str, help='[embeds,logits]', default = 'logits')
    parser.add_argument('--operations', type=str, help='Final operations performed on the forward passed data, with "x" designating no operation. Each operation yields a separate output. Any numpy axis-wide function is valid, such as mean, max, or var. Recommended to combine with other flags. For instance, mean with indices 1- and embeds mode gives the mean embedding of each sequence.', default = 'x')
    parser.add_argument('--type_mpnn', type=str, help='Specify the MPNN model type. [protein_mpnn, ligand_mpnn, per_residue_label_membrane_mpnn, global_label_membrane_mpnn, soluble_mpnn]', required=True)
    parser.add_argument('--model_checkpoint', type=str, help='Specify the MPNN model checkpoint.',required=True)
    parser.add_argument('--indices_str', type=str, help='Specify the indices of the residues to redesign.', default = '1-')
    parser.add_argument('--use_sequence', type=str, help='Specify whether to use sequence information.', default = '1')
    parser.add_argument('--as_pkl', type=str, help='Specify the name of the output pickle file.', default = '')
    parser.add_argument('--single_aa_score', type=str, help='Specify the single amino acid score.', default = '0')
    parser.add_argument('--num_workers', type=int, help='Number of workers (parallel MPNN runs).', default = 1)
    parser.add_argument('--chkpt_num', type=int, help='Number of checkpoint files to make.', default = 1)

    args = parser.parse_args()
    main(args)

