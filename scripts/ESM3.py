import torch
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.tokenization import get_esm3_model_tokenizers
from esm.pretrained import (
    ESM3_sm_open_v0,
    ESM3_structure_encoder_v0,
)
from esm.utils.structure.protein_chain import ProteinChain
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
import argparse

### CHECK FOR STRUCTURE ONLY INPUT ###

def load_model(device):
    tokenizers_pointers = get_esm3_model_tokenizers()
    tokenizers = {'sequence': tokenizers_pointers.sequence, 'structure': ESM3_structure_encoder_v0()}
    model = ESM3_sm_open_v0()
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    return model.to(device), tokenizers, device

def read_fasta(fasta_file):
    headers = []
    sequences = []
    with open(fasta_file, 'r') as inf:
        header, seq = None, None
        for line in inf:
            if line.startswith('>'):
                if header and seq:
                    headers.append(header)
                    sequences.append(seq)
                header, seq = line.strip('\n')[1:], ''
            else:
                seq += line.strip()
        if header and seq:
            headers.append(header)
            sequences.append(seq)
    return headers,sequences

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

def prep_indices_str(inputs,input_type,indices_str):
    try:
        df = read_index_file(indices_str)
        final_indices_str = []
        headers = []
        final_inputs = []
        if input_type in ['structure','both']:
            for df_index,df_item in enumerate(df['header']):
                for pointer in inputs:
                    if df_item in pointer:
                        headers.append(df_item)
                        final_inputs.append(pointer)
                        final_indices_str.append(df['indices_str'].iloc[df_index])
        elif input_type == 'sequence':
            for df_index,df_item in enumerate(df['header']):
                for inputs_index,inputs_item in enumerate(inputs[0]):
                    if inputs_item in df_item:
                        headers.append(inputs_item) # headers
                        final_inputs.append(inputs[1][inputs_index]) # sequences
                        final_indices_str.append(df['indices_str'].iloc[df_index])
        return headers,final_inputs,final_indices_str
    except FileNotFoundError:
        try:
            print('Index file not found. Parsing as a string...')
            if input_type in ['structure','both']:
                headers = [x.split('/')[-1].split('-')[1] for x in inputs]
                return headers,inputs,[indices_str]*len(inputs)
            elif input_type == 'sequence':
                return inputs[0],inputs[1],[indices_str]*len(inputs[0])
        except:
            raise ValueError(f"Index file does not exist or is not an acceptable string: {indices_str}")

def prep_names(headers, operations, model_choice, mode, input_type,indices_str,as_pkl): # generate names for output files
    operations = operations.split('_')
    all_names = {}
    if as_pkl != '':
        as_pkl = {operation: f'{model_choice}.{input_type}.{mode}.{operation}.{as_pkl}' for operation in operations}
    for operation in operations:
        if operation == "x":
            names = [f'{header}.{model_choice}.{input_type}.{mode}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
        else:
            func = getattr(np, operation, None)
            if callable(func):
                names = [f'{header}.{model_choice}.{input_type}.{mode}.{operation}.indices_{indices_str[i]}' for i,header in enumerate(headers)]
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

def perform_operations(tokenizer,outputs,mode,operations): # reorder columns for logits|probs and apply operations to outputs
    if ((mode == 'logits') | ('probs' in mode)):
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        vocab = tokenizer.get_vocab()
        new_outputs = [[] for i in range(len(outputs))] # want this to end up as a list of arrays
        for aa in amino_acids:
            aa_ls = []
            for key,item in vocab.items():
                if aa in key:
                    aa_ls.append(item)
            for i,output in enumerate(outputs):
                new_outputs[i].append(np.log(np.sum(np.exp(output[:,aa_ls]),axis=1)))
        new_outputs = [np.array(x).T for x in new_outputs]
        if (i := mode.split('_'))[0] == 'probs':
            if len(i) > 1:
                temp = float(i[1])
                new_outputs = [(np.exp(output/temp)/np.exp(output/temp).sum(axis=-1, keepdims=True)) for output in new_outputs]
            else:
                new_outputs = [np.exp(output)/np.exp(output).sum(axis=-1, keepdims=True) for output in new_outputs]
    else:
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

def process_sequences(model,tokens,indices,mode,input_type): # forward pass
    with torch.no_grad():
        if input_type == 'sequence':
            output = model.forward(sequence_tokens=tokens['sequence_tokens'])
        elif input_type == 'structure':
            output = model.forward(structure_coords=tokens['coords'],
            per_res_plddt=tokens['plddt'],
            structure_tokens=tokens['structure_tokens'])
        elif input_type == 'both':
            output = model.forward(structure_coords=tokens['coords'],
            per_res_plddt=tokens['plddt'],
            sequence_tokens=tokens['sequence_tokens'],
            structure_tokens=tokens['structure_tokens'])
        if mode == 'embeds':
            output = output.embeddings.cpu().numpy()
        elif mode == 'logits':
            output = output.sequence_logits.cpu().numpy()
    try:
        return output[0,indices]
    except:
        return None

def run(model,tokenizers,device,inputs,indices_str,mode,input_type,mlm_loop):
    outputs = []
    for i in range(len(inputs)):
        if input_type == 'sequence':
            indices = index_str_to_arrays(indices_str[i], len(inputs[i]))
            sequence_tokens = torch.tensor(tokenizers['sequence'].encode(inputs[i]), dtype=torch.int64).to(device)
            tokens = {'sequence_tokens': sequence_tokens.unsqueeze(0)}
        elif input_type in ['structure','both']:
            protein_chain = ProteinChain.from_pdb(inputs[i])
            coords, plddt, residue_index = protein_chain.to_structure_encoder_inputs()
            _, structure_tokens = tokenizers['structure'].encode(coords=coords, residue_index=residue_index)
            indices = index_str_to_arrays(indices_str[i], structure_tokens.shape[1])
            # Add BOS/EOS padding to structure tokens
            coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=torch.inf).to(device)
            plddt = F.pad(plddt, (1, 1), value=0).to(device)
            structure_tokens = F.pad(structure_tokens, (1, 1), value=0).to(device)
            structure_tokens[:, 0] = 4098
            structure_tokens[:, -1] = 4097
            tokens = {'coords': coords, 'plddt': plddt, 'structure_tokens': structure_tokens}
            if input_type == 'both':
                sequence_tokens = torch.tensor(tokenizers['sequence'].encode(protein_chain.sequence), dtype=torch.int64).to(device)
                tokens['sequence_tokens'] = sequence_tokens.unsqueeze(0)
        if mlm_loop: ### CHECK FOR STRUCTURE ONLY INPUT ###
            output = []
            base_sequence_tokens = tokens['sequence_tokens']
            try:
                for index in indices:
                    sequence_tokens = base_sequence_tokens.clone()
                    sequence_tokens[0,index] = 32 # mask token
                    tokens['sequence_tokens'] = sequence_tokens
                    output.append(process_sequences(model,tokens,index,mode,input_type))
                output = np.vstack(output)
            except:
                output = None
        else:
            try:
                output = process_sequences(model,tokens,indices,mode,input_type)
            except:
                output = None
        outputs.append(output)
    return outputs

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
    input_pointer = args.input_pointer
    input_type = args.input_type
    device = args.device
    output_dir = args.output_dir
    indices_str = args.indices
    mode = args.mode
    operations = args.operations
    mlm_loop = args.mlm_loop
    as_pkl = args.as_pkl
    chkpt_num = args.chkpt_num


    if input_type == 'sequence':
        headers,sequences = read_fasta(input_pointer)
        inputs = [headers,sequences]
    elif input_type in ['structure','both']:
        if os.path.isfile(input_pointer):
            inputs = [input_pointer]
        elif os.path.isdir(input_pointer):
            inputs = [os.path.join(input_pointer, f) for f in os.listdir(input_pointer)]
    headers,inputs,indices_str = prep_indices_str(inputs,input_type,indices_str)
    model, tokenizers, device = load_model(device)
    for i in range(chkpt_num):
        if chkpt_num == 1: # if only one checkpoint...
            s = 0
            e = len(headers)
            chkpt_str = ''
        elif i == chkpt_num-1:
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
        outputs = run(model,tokenizers,device,inputs_chunk,indices_str_chunk,mode,input_type,mlm_loop)
        pop_j = []
        for j,output in enumerate(outputs):
            if output is None:
                #print(headers_chunk[j])
                pop_j.append(j)
        print(f'len(pop_j): {len(pop_j)}')
        for j in pop_j[::-1]:
            try:
                headers_chunk.pop(j)
                inputs_chunk.pop(j)
                indices_str_chunk.pop(j)
                outputs.pop(j)
            except:
                print(f'Error popping index {j}.')
                continue
        
        outputs = perform_operations(tokenizers['sequence'],outputs,mode,operations)
        model_choice = 'ESM3_sm_open_v0'
        names,as_pkl_chunk = prep_names(headers_chunk, operations, model_choice, mode, input_type, indices_str_chunk, as_pkl+chkpt_str)
        save_outputs(outputs, output_dir, names, as_pkl_chunk)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process sequences using ESM-3 model.')
    parser.add_argument('--input_pointer', type=str, help='Pointer to input. Either a pointer to a fasta file, a pdb file, or a directory of pdb files.')
    parser.add_argument('--input_type', type=str, help='Text input specifying the input to ESM3. Options, ["structure","sequence","both"].')
    parser.add_argument('--device', type=str, help='Specify the Cuda device index',default = '0')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--indices', type=str, help='Tab delimited text file indicating which positions on each protein to run. Alternatively, a string for the same position in all proteins.',default = '1-')
    parser.add_argument('--mode', type=str, help='Mode to run model. Options [embeds, logits, probs]. Sampling temp can be specified by appending an integer between 0 and 99 as in --mode probs_24 for temp = 0.24.', default = 'logits')
    parser.add_argument('--operations', type=str, help='Final operations performed on the forward passed data, with "x" designating no operation. Each operation yields a separate output. Any numpy axis-wide function is valid, such as mean, max, or var. Recommended to combine with other flags. For instance, mean with indices 1- and embeds mode gives the mean embedding of each sequence.', default = 'x')
    parser.add_argument('--mlm_loop', action='store_true', help='Used with mode=[logits||probs]. Selected indices will be iteratively masked prior to running. Default: False')
    parser.add_argument('--as_pkl', type=str, help='Save output as a single pickle file rather than as individual *.npy files. Give output filename prefix to activate string.', default = '')
    parser.add_argument('--chkpt_num', type=int, help='Number of checkpoints to run', default = 1)

    args = parser.parse_args()
    main(args)
