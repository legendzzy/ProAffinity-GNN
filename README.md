Data preparation:
1. prepare PDBQT file, saved under data/pdbqt/
   When having PDB file, install ADFR and use command  prepare_receptor -r ****.pdb -A hydrogens your_pdbqt_file.pdbqt 
3. prepare FASTA file, saved under data/FASTA/mixed, also put FASTA file in 2mole or 3mole directory according to the molecule number this pdb contains.
4. prepare 2 PPI data index files, one contains the pairwise division for pdb, and another contains the binding affinity value (pKa).

Graph construct:
1. run pdb2graph.py and pdb2graph_individual.py, the intermediate files generated will be saved in /data/graph_construct/inter_graph and /data/graph_construct/individual_graph respectively.
2. run graph_construct.py and graph_construct_indi.py, the graph data will be saved in /data/graph/inter/graph and /data/graph/individual_graph respectively.

Train and test:
Run ProAffinity_GNN.py to train. To retrain this work, please prepare the full training dataset.
Run test.py to test. 
Please prepare the data needed to be trained or tested.

Prerequisite:
python=3.8
pytorch=2.2
pytorch-geometric=2.30
transformers=4.38 (pip install)
scikit-learn=1.3.2
