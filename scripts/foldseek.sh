#!/usr/bin/env bash
set -euo pipefail

##############################################
# Generate a FASTA file of Foldseek tokens
# from a directory of PDB structures.
#
# Usage:
#   generate_foldseek_fasta.sh <output_dir> <structure_db_dir>
#
# Arguments:
#   output_dir        Directory to store generated Foldseek DB files
#   structure_db_dir  Directory containing PDB files (input)
##############################################

# ---- Argument Parsing ----
if [[ $# -ne 2 ]]; then
    echo "Error: Wrong number of arguments."
    echo "Usage: $0 <output_dir> <structure_db_dir>"
    exit 1
fi

new_dir="$1"
structure_db="$2"

echo "Output directory:        $new_dir"
echo "Structure database dir:  $structure_db"
echo
echo "[1/4] Creating output directory..."
mkdir -p "$new_dir"
echo "[2/4] Running: foldseek createdb"
foldseek createdb "$structure_db" "$new_dir/foldseek_db"
echo "[3/4] Running: foldseek lndb"
foldseek lndb "$new_dir/foldseek_db_h" "$new_dir/foldseek_db_ss_h"
echo "[4/4] Running: foldseek convert2fasta"
foldseek convert2fasta "$new_dir/foldseek_db_ss" "$new_dir/foldseek_db_ss.fasta"
cp "$new_dir/foldseek_db_ss.fasta" "$structure_db/foldseek.fasta"

echo
echo "Done! FASTA written to:"
echo "  $structure_db/foldseek.fasta"
