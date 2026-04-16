import re
import argparse


def read_fasta(fasta_file):
    sequences = []
    headers = []
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
    headers = [x.split('|')[1] for x in headers]
    return headers, sequences


def write_indices(headers, indices, filename):
    with open(filename, 'w') as out:
        for i,j in enumerate(indices):
            if j != '':
                out.write(headers[i] + '\t' + j + '\n')
    return


def main(args):
    pattern = args.pattern
    fasta = args.fasta
    outf = args.outf
    headers,sequences = read_fasta(fasta)
    get_indices = lambda x,pattern: '_'.join([str(i.start()+1) for i in re.finditer(pattern,x)])
    x_indices = [get_indices(x, pattern) for x in sequences]
    write_indices(headers, x_indices, f'{outf}.txt')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prep indices in a fasta file using regex patterns.')
    parser.add_argument('--fasta', type=str, help='Path to the fasta file')
    parser.add_argument('--pattern', type=str, help='Regex pattern')
    parser.add_argument('--outf',type=str, help='output filename. ".txt" will be appended by the program')
    args = parser.parse_args()
    main(args)
