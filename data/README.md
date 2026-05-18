# Dataset Schema Reference Guide

This repository contains curated datasets for protein engineering, variant effect prediction, and deep mutational scanning (DMS). This README provides a comprehensive reference guide for the data fields across three primary files.

---

## 1. FireProtDB_Datasets.csv
This dataset profiles structural and thermodynamic stability data, mapping mutation landscapes to specific positions and architectural attributes of proteins.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `uniprot_id` | String | Unique identifier for the protein record in the UniProt Knowledgebase. |
| `protein_name` | String | Standard descriptive name of the protein. |
| `species` | String | Biological source organism from which the protein originates. |
| `ec_number` | String | Enzyme Commission number detailing the catalytic activity, if applicable. |
| `pdb_id` | String | Four-character identifier for the corresponding 3D structure in the Protein Data Bank. |
| `chain` | String | Specific macromolecular chain identifier within the structural PDB file. |
| `interpro_families` | String | Functional classifications and signature families mapped via InterPro. |
| `sequence_length` | Integer | Total count of amino acid residues in the canonical protein sequence. |
| `sequence` | String | Full wild-type amino acid sequence in single-letter code format. |
| `unique_mutation_count` | Integer | Total number of distinct amino acid substitutions evaluated for this protein. |
| `unique_position_count` | Integer | Total number of individual residue positions where mutations were introduced. |
| `total_experiment_count` | Integer | Aggregate count of experimental assays or stability measurements recorded. |
| `mean_dms_score` | Float | The averaged fitness, activity, or stability score across all evaluated variants. |
| `min_position` | Integer | The starting sequence coordinate of the experimentally monitored region. |
| `max_position` | Integer | The ending sequence coordinate of the experimentally monitored region. |
| `datasets` | String | Reference tags indicating which specific sub-studies or source pipelines include this record. |

---

## 2. ProteinGym_Datasets.csv
This dataset tracks comprehensive deep mutational scanning (DMS) records, detailing sequence fitness landscapes alongside detailed Multiple Sequence Alignment (MSA) metadata for machine learning evaluation.

### Core Meta & Source Identifiers
* `DMS_id`: Unique string identifier for the deep mutational scanning assay.
* `DMS_filename`: Name of the file containing the underlying raw mutational data scores.
* `UniProt_ID`: Corresponding identifier in the UniProt database.
* `taxon`: Taxonomic category of the source organism (e.g., eukaryote, prokaryote, virus).
* `source_organism`: Scientific name of the organism from which the protein was derived.
* `molecule_name`: Specific biochemical or functional name of the target molecule.

### Sequence & Mutation Profiles
* `target_seq`: Single-letter amino acid sequence used as the baseline wild-type in the assay.
* `seq_len`: Total character length of the target sequence.
* `includes_multiple_mutants`: Boolean indicator (True/False) showing if combinations of multiple amino acid substitutions are evaluated.
* `DMS_total_number_mutants`: Total count of all variant sequences evaluated within the assay.
* `DMS_number_single_mutants`: Total count of individual single-point mutation variants.
* `DMS_number_multiple_mutants`: Total count of complex variants containing more than one mutation.
* `region_mutated`: Text description or coordinates of the domain or area subjected to mutagenesis.

### Assay & Binarization Criteria
* `DMS_binarization_cutoff`: Numerical threshold applied to split continuous assay scores into discrete functional or non-functional classes.
* `DMS_binarization_method`: Mathematical or programmatic approach utilized to determine the cutoff threshold.
* `selection_assay`: Experimental methodology leveraged to measure variant performance.
* `selection_type`: Classification of the assay approach (e.g., organismal fitness, binding affinity, enzyme activity).
* `coarse_selection_type`: Broad classification category for the underlying experimental selection mechanism.

### Publication & Literature Mapping
* `first_author`: Surname of the primary author of the source study.
* `title`: Full title of the peer-reviewed publication or preprint detailing the dataset.
* `year`: Publication calendar year.
* `jo`: Abbreviated journal title where the study was published.

### Alignment, Structure & Provenance Attributes
* `MSA_filename`: File name of the sequence alignment used for evolutionary modeling.
* `raw_DMS_directionality`: Factor indicating if higher numerical scores represent increased fitness (+1) or decreased fitness (-1).
* `raw_DMS_mutant_column`: Header name of the column containing the variant formatting designations in the raw file.
* `weight_file_name`: Reference to the sequence weight file generated during MSA processing for modeling corrections.
* `pdb_file`: Corresponding structural definition file from the Protein Data Bank used for spatial analysis.
* `pdb_range`: Exact sequence coordinate bounds covered by the structural PDB coordinates.
* `ProteinGym_version`: Version label of the ProteinGym benchmark pipeline this file maps to.
* `raw_mut_offset`: Integer alignment shift offset value applied to align raw experimental sequence coordinates with standard reference models.

---

## 3. Other_Datasets.csv
An auxiliary collection indexing comparative variant datasets, general mutations, and literature links.

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `Protein Name` | String | Common structural or functional name of the protein. |
| `UniProt ID` | String | Unique index matching the canonical record within the UniProt database. |
| `Category` | String | Functional grouping or project category assigned to the record. |
| `Mut. overlap` | String/Float | Measure of shared variant spaces or intersection with other primary files. |
| `Mutations` | String | Summary array or list format of specific variant definitions explored. |
| `Citation` | String | Reference details mapping the record back to its primary literature source. |