from transformers import T5Tokenizer, AutoModelForSeq2SeqLM, T5EncoderModel
import torch
import numpy as np
import argparse
import pandas as pd
import pickle
import os

def load_model(device,mode):
    print('Loading ProstT5 model...')
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    if mode == 'embeds':
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained("Rostlab/ProstT5").to(device)
    model.full() if device=='cpu' else model.half()
    return model, tokenizer, device

def read_fasta(fasta_file,isfoldseek=False):
    headers = []
    sequences = []
    with open(fasta_file, 'r') as inf:
        header, seq = None, None
        for line in inf:
            if line.startswith('>'):
                if header and seq:
                    headers.append(header)
                    sequences.append(" ".join(list(seq)))
                header, seq = line.strip('\n')[1:], ''
            else:
                seq += line.strip()
        if header and seq:
            headers.append(header)
            sequences.append(" ".join(list(seq)))
    if isfoldseek==True: #### CHANGE FOR PRODUCTION
        sequences = ['<fold2AA> '+x.lower() for x in sequences]
    else: #### CHANGE FOR PRODUCTION
        sequences = ['<AA2fold> '+x for x in sequences]
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

def prep_indices_str(indices_str,headers,sequences):
    try:
        df = read_index_file(indices_str)
        final_headers = []
        final_sequences = []
        final_indices_str = []
        for i,seq in enumerate(sequences):
            df_subset = df[df['header'] == headers[i]]
            if headers[i] in df_subset['header'].values:
                final_indices_str.append(df_subset['indices_str'].values[0])
                final_headers.append(headers[i])
                final_sequences.append(seq)
        return final_headers,final_sequences,final_indices_str
    except FileNotFoundError:
        try:
            print('Index file not found. Parsing as a string...')
            return headers,sequences,[indices_str]*len(sequences)
        except:
            raise ValueError(f"Index file does not exist or is not an acceptable string: {indices_str}")

def sort_by_length(headers,sequences,indices_str):
    enumerated_array = list(enumerate(sequences))
    sorted_array = sorted(enumerated_array, key=lambda x: len(x[1]))
    order = [item[0] for item in sorted_array]
    headers = [headers[i] for i in order]
    sequences = [sequences[i] for i in order]
    indices_str = [indices_str[i] for i in order]
    return headers,sequences,indices_str

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

def sequence_masking(sequences, indices):
    masked_sequences = []
    for i,sequence in enumerate(sequences):
        masked_sequence = [f'{sequence[:9+j*2-1]}<extra_id_0> {sequence[9+j*2+1:]}' for j in indices[i]]
        masked_sequences.append(masked_sequence)
    return masked_sequences

def prep_mlm(sequences,indices):
    all_sequences = []
    all_indices = []
    masked_sequences = sequence_masking(sequences,indices)
    for i,seq in enumerate(masked_sequences):
        for j in range(len(seq)):
            all_sequences.append(seq[j])
            all_indices.append(indices[i][j])
    return all_sequences, all_indices

def perform_operations(tokenizer,outputs,mode,operations,isfoldseek): # reorder columns for logits|probs and apply operations to outputs
    if ((mode == 'logits') | ('probs' in mode)):
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        if isfoldseek==False:
            amino_acids = [x.lower() for x in amino_acids]
        vocab = tokenizer.get_vocab()
        new_outputs = [[] for i in range(len(outputs))] # want this to end up as a list of arrays
        for aa in amino_acids:
            aa_ls = []
            for key,item in vocab.items():
                if (aa in key) & ('>' not in key):
                    aa_ls.append(item)
            for i,output in enumerate(outputs):
                if output is not None:
                    new_outputs[i].append(np.log(np.sum(np.exp(output[:,aa_ls]),axis=1)))
                else:
                    new_outputs[i].append(None)
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

def process_sequences(model,tokenizer,device,sequences,indices_str,mode): # forward pass, subset indices, perform operations
    outputs = []
    if mode == 'embeds':
        ids = tokenizer.batch_encode_plus(sequences,
                                        add_special_tokens=True,
                                        padding="longest",
                                        return_tensors='pt').to(device)
        with torch.no_grad():
            output = model(
                    ids.input_ids, 
                    attention_mask=ids.attention_mask
                    ).last_hidden_state.cpu().numpy().astype(np.float128)
        return [output[i,j] for i, j in enumerate(indices_str)]
    else:    
        for i, sequence in enumerate(sequences):
            indices = index_str_to_arrays(indices_str[i], len(sequence.split(' '))-1)
            tokens = tokenizer.batch_encode_plus([sequence],add_special_tokens=True,padding="longest",return_tensors='pt').to(device)
            len_seq = len(sequence.split(' '))
            try:
                with torch.no_grad():
                    if mode in ['logits','probs']:
                        output = model.generate(tokens.input_ids,attention_mask=tokens.attention_mask, 
                        max_length=len_seq,
                        min_length=len_seq,
                        output_scores=True,
                        return_dict_in_generate=True)
                        outputs.append(torch.cat(output.scores).cpu().numpy().astype(np.float128)[[x-1 for x in indices]])
                    elif mode == 'embeds':
                        output = model(tokens['input_ids'],
                                        attention_mask=tokens['attention_mask'],
                                        decoder_input_ids=tokens['input_ids'],
                                        output_hidden_states=True)
                        outputs.append(output['encoder_last_hidden_state'].cpu().numpy().astype(np.float128)[indices])
            except:
                outputs.append(None)
    return outputs

def batch_run(model, tokenizer, sequences, indices_str, device, batch_size, mode, operations, isfoldseek):
    sequence_index = 0
    batch_index = 0
    outputs = {x: [] for x in operations.split('_')}
    while sequence_index < len(sequences):
        try:
            batch_sequences = sequences[sequence_index:sequence_index+batch_size]
            batch_indices = [index_str_to_arrays(index, len(seq.split(' '))-1) for index,seq in zip(indices_str[sequence_index:sequence_index+batch_size],batch_sequences)]
            batch_outputs = process_sequences(model,tokenizer,device,batch_sequences,batch_indices,mode)
            batch_outputs = perform_operations(tokenizer,batch_outputs,mode,operations,isfoldseek) # {operation: [output1, output2, ...]}
            for key in batch_outputs:
                outputs[key] += batch_outputs[key]
            batch_index += 1
            sequence_index += batch_size
            print(f'{sequence_index} of {len(sequences)} complete')
        except RuntimeError as error:
            batch_size = batch_size // 2
            print(f'Batch size too large at batch no {batch_index}. Reducing to: {batch_size}')
        except IndexError as error:
            batch_sequences = sequences[sequence_index:sequence_index+batch_size]
            batch_indices = [index_str_to_arrays(index, len(seq.split(' '))-1) for index,seq in zip(indices_str[sequence_index:sequence_index+batch_size],batch_sequences)]
            batch_outputs = process_sequences(model,tokenizer,device,batch_sequences,batch_indices,mode)
            batch_outputs = perform_operations(tokenizer,batch_outputs,mode,operations,isfoldseek) # {operation: [output1, output2, ...]}
            for key in batch_outputs:
                outputs[key] += batch_outputs[key]
            break
    return outputs, batch_size # {operation: [output1, output2, ...]}, batch_size

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
    fasta = args.fasta
    isfoldseek = args.isfoldseek
    device = args.device
    output_dir = args.output_dir
    indices_str = args.indices
    mode = args.mode
    operations = args.operations
    as_pkl = args.as_pkl
    chkpt_num = args.chkpt_num

    headers,sequences = read_fasta(fasta,isfoldseek)
    headers,sequences,indices_str = prep_indices_str(indices_str,headers,sequences)
    headers,sequences,indices_str = sort_by_length(headers,sequences,indices_str)
    model_choice='ProstT5'
    model, tokenizer, device = load_model(device,mode)
    batch_size = 1024
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
        sequences_chunk = sequences[s:e]
        indices_str_chunk = indices_str[s:e]
        if mode=='embeds':
            outputs,batch_size = batch_run(model,tokenizer,sequences_chunk,indices_str_chunk,device,batch_size,mode,operations,isfoldseek)
        else:
            outputs = process_sequences(model,tokenizer,device,sequences_chunk,indices_str_chunk,mode)
            outputs = perform_operations(tokenizer,outputs,mode,'x',isfoldseek)
        pop_j = []
        for j,output in enumerate(list(outputs.values())[0]):
            if output is None:
                pop_j.append(j)
        print(f'len(pop_j): {len(pop_j)}')
        for j in pop_j[::-1]:
            try:
                headers_chunk.pop(j)
                sequences_chunk.pop(j)
                indices_str_chunk.pop(j)
                for operation in outputs:
                    outputs[operation].pop(j)
            except:
                print(f'Error popping index {j}.')
                continue
        names,as_pkl_dict = prep_names(headers_chunk, operations, model_choice, mode, indices_str_chunk, as_pkl+chkpt_str)
        save_outputs(outputs,output_dir,names,as_pkl_dict)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inverse fold 3Di sequences using ProstT5 model.')
    parser.add_argument('--fasta', type=str, help='Path to the amino acid fasta file')
    parser.add_argument('--isfoldseek', action='store_true', help='Flag to specify that Fasta file is from FoldSeek')
    parser.add_argument('--device', type=str, help='Specify the Cuda device index',default = '0')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--indices', type=str, help='Tab delimited text file indicating which positions on each protein to run. Alternatively, a string for the same position in all proteins.',default = '1-')
    parser.add_argument('--mode', type=str, help='Mode to run model. Options [embeds, logits, probs]. Sampling temp can be specified by appending an integer between 0 and 99 as in --mode probs_24 for temp = 0.24.', default = 'logits')
    parser.add_argument('--operations', type=str, help='Final operations performed on the forward passed data, with "x" designating no operation. Each operation yields a separate output. Any numpy axis-wide function is valid, such as mean, max, or var. Recommended to combine with other flags. For instance, mean with indices 1- and embeds mode gives the mean embedding of each sequence.', default = 'x')
    parser.add_argument('--as_pkl', type=str, help='Save output as a single pickle file rather than as individual *.npy files. Give output filename prefix to activate string.', default = '')
    parser.add_argument('--chkpt_num', type=int, help='Number of checkpoint files to make.', default = 1)

    args = parser.parse_args()
    main(args)
