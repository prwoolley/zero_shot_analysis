# Dataset Reference Guide

This directory contains the three dataset summary files used throughout the manuscript, grouped by benchmarking role. This README standardizes the field descriptions for the FireProtDB thermostability benchmark, the ProteinGym fitness benchmark, and the curated comparison sets used for inter-experimenter, inter-phenotype, and new-to-nature analyses.

---

## 1. FireProtDB Thermostability Benchmark (`FireProtDB_Datasets.csv`)
This file summarizes the FireProtDB-derived thermostability datasets used in the manuscript's stability benchmark analyses.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `uniprot_id` | String | UniProt identifier for the protein sequence used for the record. |
| `protein_name` | String | Protein name associated with the UniProt entry. |
| `species` | String | Source organism for the protein. |
| `ec_number` | String | Enzyme Commission number when the protein is an annotated enzyme. |
| `pdb_id` | String | One or more PDB structure identifiers associated with the protein. |
| `chain` | String | Chain identifier used for the structure record. |
| `interpro_families` | String | InterPro family or domain annotations linked to the protein. |
| `sequence_length` | Integer | Length of the protein sequence in amino acids. |
| `sequence` | String | Wild-type amino acid sequence. |
| `unique_mutation_count` | Integer | Number of unique single amino-acid substitutions compiled for the protein. |
| `unique_position_count` | Integer | Number of sequence positions represented by those mutations. |
| `total_experiment_count` | Integer | Total number of experimental measurements aggregated for the protein. |
| `mean_dms_score` | Float | Mean stability-related score across the compiled mutation set. |
| `min_position` | Integer | Minimum residue position covered by the compiled mutations. |
| `max_position` | Integer | Maximum residue position covered by the compiled mutations. |
| `datasets` | String | Source dataset identifiers contributing measurements for the protein. |

---

## 2. ProteinGym Fitness Benchmark (`ProteinGym_Datasets.csv`)
This file summarizes the ProteinGym deep mutational scanning datasets used for the manuscript's fitness benchmark and related comparison analyses.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `DMS_id` | String | Unique identifier for the deep mutational scanning experiment. |
| `DMS_filename` | String | Filename for the raw DMS score table associated with the experiment. |
| `UniProt_ID` | String | UniProt identifier for the assayed protein. |
| `taxon` | String | Broad taxonomic grouping of the source organism. |
| `source_organism` | String | Source organism from which the assayed protein was derived. |
| `target_seq` | String | Wild-type amino acid sequence used as the assay reference. |
| `seq_len` | Integer | Length of the target sequence in amino acids. |
| `includes_multiple_mutants` | Boolean | Indicates whether the full dataset includes multi-mutant variants in addition to single mutants. |
| `DMS_total_number_mutants` | Integer | Total number of mutant sequences reported for the experiment. |
| `DMS_number_single_mutants` | Integer | Number of single amino-acid substitution variants in the experiment. |
| `DMS_number_multiple_mutants` | Integer | Number of multi-mutant variants in the experiment. |
| `DMS_binarization_cutoff` | Float | Threshold used to binarize continuous DMS scores when a binary label is needed. |
| `DMS_binarization_method` | String | Method used to define the binarization cutoff. |
| `first_author` | String | First author of the source study. |
| `title` | String | Title of the source article or preprint. |
| `year` | Integer | Publication year of the source study. |
| `jo` | String | Journal name, abbreviated citation string, or publication venue identifier. |
| `region_mutated` | String | Sequence region or coordinates covered by mutagenesis in the experiment. |
| `molecule_name` | String | Protein or molecule name used in the source dataset. |
| `selection_assay` | String | Experimental assay used to measure mutational effects. |
| `selection_type` | String | Specific phenotype or assay readout measured in the experiment. |
| `MSA_filename` | String | Filename of the multiple sequence alignment used for evolutionary-context analyses. |
| `raw_DMS_directionality` | Integer | Directionality flag indicating whether larger raw scores correspond to improved or reduced function. |
| `raw_DMS_mutant_column` | String | Column name in the raw DMS file containing the mutation identifiers. |
| `weight_file_name` | String | Filename for the MSA-derived sequence weighting file. |
| `pdb_file` | String | Structure file associated with the experiment. |
| `pdb_range` | String | Residue range in the structure file aligned to the assayed sequence. |
| `ProteinGym_version` | String | ProteinGym release version associated with the record. |
| `raw_mut_offset` | Integer | Residue index offset used to align raw mutation numbering to the reference sequence. |
| `coarse_selection_type` | String | Higher-level phenotype category used to group related assay types. |

---

## 3. Curated Comparison Sets (`Other_Datasets.csv`)
This file lists the curated proteins used for the manuscript's inter-experimenter, inter-phenotype, and new-to-nature comparisons.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `Protein Name` | String | Common name of the protein used in the comparison set. |
| `UniProt ID` | String | UniProt identifier for the wild-type protein sequence used in the analysis. |
| `Category` | String | Comparison category: `inter-experimenter`, `inter-phenotype`, or `new-to-nature`. |
| `Mut. overlap` | String/Integer | Number of overlapping mutations across matched datasets for the protein, or `-` when overlap is not applicable. |
| `Mutations` | String | Number of mutations reported in each contributing dataset or experiment. |
| `Citation` | String | Formatted literature references for the studies contributing the data. |