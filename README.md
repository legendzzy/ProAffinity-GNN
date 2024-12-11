# ProAffinity-GNN

## Data Preparation

### 1. Prepare PDBQT File
- Save under `data/pdbqt/`.
- If you have a PDB file, install ADFR and use the command:
  `prepare_receptor -r ****.pdb -A hydrogens -o your_pdbqt_file.pdbqt`


### 2. Prepare FASTA File
- Save under `data/FASTA/mixed`. Also, place the FASTA file in the `2mole` or `3mole` directory according to the molecule number the pdb contains.

### 3. Prepare PPI Data Index Files
- One file contains the pairwise division for pdb, and another contains the binding affinity value (pKa).

## Graph Construction

### 1. Run `pdb2graph.py` and `pdb2graph_individual.py`
- The intermediate files generated will be saved in `/data/graph_construct/inter_graph` and `/data/graph_construct/individual_graph` respectively.

### 2. Run `graph_construct.py` and `graph_construct_indi.py`
- The graph data will be saved in `/data/graph/inter/graph` and `/data/graph/individual_graph` respectively.

## Train and Test

- Run `ProAffinity_GNN.py` to train. To retrain this work, please prepare the full training dataset.
- Run `test.py` to test.
- Please prepare the data needed to be trained or tested. The example data files are under the `data` folder.
- The model trained in this work in provided in the `model` folder.

## Inference

- To easily use ProAffinity-GNN to test on any protein-protein complex, we provide the inference code, see `/ProAffinity-GNN_inference`.
- Run `python ProAffinity-GNN_inference.py -f [your pdbqt input file] -c [interaction chains, e.g., AB,C]`
- Example: Run `cd ./ProAffinity-GNN_inference`, and Run `python ProAffinity-GNN_inference.py -f 1ak4_processed.pdbqt -c A,D`

## Prerequisites

- Python=3.8
- PyTorch=2.2
- PyTorch-Geometric=2.30
- Transformers=4.38 (Install using pip)
- Scikit-Learn=1.3.2

