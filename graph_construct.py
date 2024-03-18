from transformers import AutoTokenizer, EsmModel
import torch
import copy
import math
import pickle
import os
import numpy as np
from torch_geometric.data import Data

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model.eval()

def get_fasta_seq(pdb_tuple):
    fastaA = pdb_tuple[0]
    fastaB = pdb_tuple[1]
    return fastaA, fastaB

path = 'data/graph_construct/inter_graph/'
filenames = os.listdir(path)

def get_distance(x1, y1, z1, x2, y2, z2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    return distance

def read_y(filename):
    y_dict = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pdb = line.split()[0].lower()
            y = float(line.split()[1])
            y = round(y, 2)
            y_dict.update({pdb:y})
    return y_dict

atom_type = ['A', 'C', 'OA', 'N', 'NA' 'SA', 'HD']

atom_pair = ['A_A', 'A_C', 'A_OA', 'A_N', 'A_NA', 'A_SA', 'A_HD', 
            'C_C', 'C_OA', 'C_N', 'C_NA', 'C_SA', 'C_HD',
            'OA_OA', 'OA_N', 'OA_NA', 'OA_SA', 'OA_HD',
            'N_N', 'N_NA', 'N_SA', 'N_HD', 
            'NA_NA', 'NA_SA', 'NA_HD',
            'SA_SA', 'SA_HD',
            'HD_HD']

bin_number = 10
type_number = len(atom_pair)
inter_distance = 15
y_dict = read_y('/data/a/zhiyuan/dataset/PP/PPIdataindex.txt')


for pdb in filenames:
    print(pdb)

    info = pickle.load(open('data/graph_construct/inter_graph/' + pdb, 'rb'))

    fasta1_list, fasta2_list = get_fasta_seq(info[2])
    output1_list = []
    output2_list = []

    for fasta in fasta1_list:
        input1 = tokenizer(fasta, return_tensors="pt")
        output1 = model(**input1)
        last_hidden_state1 = output1.last_hidden_state
        last_hidden_state1 = torch.squeeze(last_hidden_state1)
        # get the token from the 2nd to the 2nd last one
        last_hidden_state1 = last_hidden_state1[1:-1]
        output1_list.append(last_hidden_state1)


    for fasta in fasta2_list:
        input2 = tokenizer(fasta, return_tensors="pt")
        output2 = model(**input2)
        last_hidden_state2 = output2.last_hidden_state
        last_hidden_state2 = torch.squeeze(last_hidden_state2)
        # get the token from the 2nd to the 2nd last one
        last_hidden_state2 = last_hidden_state2[1:-1]
        output2_list.append(last_hidden_state2)


    x1 = torch.cat(output1_list, 0)
    x2 = torch.cat(output2_list, 0)
    x = torch.cat((x1, x2), 0)

    pairs = info[0]
    edge_index = info[1]

    try:
        edge_feature = []
        for pair in pairs:

            edge_encoding = np.zeros(type_number * bin_number)
            residueA = pair[0]
            residueB = pair[1]

            for atom1 in residueA['atoms']:
                x1 = float(atom1['x'])
                y1 = float(atom1['y'])
                z1 = float(atom1['z'])
                type1 = atom1['pdbqt_type']

                for atom2 in residueB['atoms']:
                    x2 = float(atom2['x'])
                    y2 = float(atom2['y'])
                    z2 = float(atom2['z'])
                    type2 = atom2['pdbqt_type']

                    dis = get_distance(x1, y1, z1, x2, y2, z2)

                    bin_n = math.ceil(dis / (inter_distance / bin_number))

                    if bin_n > 10:
                        bin_n = 10

                    if type1 + '_' + type2 in atom_pair:
                        pair_type = type1 + '_' + type2       
                    elif type2 + '_' + type1 in atom_pair:
                        pair_type = type2 + '_' + type1
                    else:
                        print('no match type!' + 'type1:' + type1 + 'type2:' + type2)
                        pair_type = 'others'
                        
                    pair_type_index = atom_pair.index(pair_type)
                    encoding_index = (bin_n - 1) * type_number + pair_type_index
                    edge_encoding[encoding_index] = edge_encoding[encoding_index] + 1

            edge_feature.append(torch.from_numpy(edge_encoding))                

        edge_feature = torch.stack(edge_feature, 0) 
        edge_feature = torch.cat((edge_feature, edge_feature), 0)

    except Exception as e:
        print(e)
        continue

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_feature, y=y_dict[pdb])
    save_path = 'data/graph/inter_graph/' + pdb

    with open(save_path, 'wb') as f_save:
        pickle.dump(data, f_save)
    