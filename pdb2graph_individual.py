import copy
import math
import torch
import pickle
import os
import re
import itertools

def only_letters(s):
    return re.sub('[^a-zA-Z]', '', s)

# return a tuple of (pdb_list, chain_list), where each element in the chain_list is a 2-element list of chains of protein part and ligand part
# for the corresponding pdb

def get_chainlist_from_indexfile(chainindex):
    pdb_list = []
    chain_list = []
    with open(chainindex, 'r') as f:
        lines = f.readlines()  
        for line in lines:
            pdb = line.split('\t')[0].strip()
            # pdb = pdb.split('.')[0]
            pdb_list.append(pdb)
            chains = line.split('\t')[1].strip()
            chains = chains.split(';')
            while '' in chains:
                chains.remove('')

            for i, c in enumerate(chains):
                chains[i] = only_letters(c).strip()
            chain_list.append(chains)
            
    return (pdb_list, chain_list)

def find_first_numeric_part(s):
    # Find the first sequence of digits in the string
    match = re.search(r'\d+', s)
    # Return the matched part if found, otherwise return None
    return match.group(0) if match else None

def check_res_number(residuelist):
    char_index = []
    for i, res in enumerate(residuelist):
        if res['number'].isnumeric() == False:
            char_index.append(i)
    if len(char_index) == 0:
        return
        
    list_len = len(residuelist) 
    if(char_index[-1] == list_len - 1):
        last_num = find_first_numeric_part(residuelist[-1]['number'])
        residuelist[-1]['number'] = last_num

    for i in reversed(char_index):
        if i == list_len - 1:
            continue
        
        residuelist[i]['number'] = residuelist[i+1]['number']
        for j in range(i+1, len(residuelist)):
            residuelist[j]['number'] = int(residuelist[j]['number']) + 1

# get one part of the protein-protein complex from the pdb file,
# divide this one part by chains, so that we can handle this one part as several chains, forming a list

def get_residue_list_from_file(filename, chain_list):
    
    residue_chain_list = []
    try: 
   
        with open(filename, 'r') as f:

            lines = f.readlines()

            for chain in chain_list:
                previous_res = -1
                previous_res_type = ''
                atomlist = []
                atom = {}       
                residue = {}
                current_res_chain = ''
                residuelist = []

                for line in lines:
                    atomline = line.split()
                    if atomline[0].strip() != 'ATOM' and atomline[0].strip() != 'TER':
                        continue

                    if line[21] != chain:
                        continue

                    if atomline[0] == 'TER':
                        residue['type'] = aminoacid_abbr[previous_res_type]
                        residue['number'] = previous_res
                        residue['atoms'] = copy.deepcopy(atomlist)
                        residue['chain'] = current_res_chain

                        has_CA = False
                        for atomCA in residue['atoms']:
                            if atomCA['type'] == 'CA':
                                has_CA = True
                                break
                        if has_CA == True:
                            residuelist.append(copy.deepcopy(residue))
                        else:
                            residue.clear()

                        previous_res = -1 # reset previous_res to init
                        continue

                    atom['pdbqt_type'] = line[77:79].strip()
                    atom['type'] = line[12:16].strip()
                    atom['x'] = line[30:38].strip()
                    atom['y'] = line[38:46].strip()
                    atom['z'] = line[46:54].strip()

                    current_res = line[22:27].strip() # residue number
                    current_res_type = line[17:21].strip() # residue type
                    current_res_chain = line[21] # residue chain

                    if current_res != previous_res and previous_res != -1: # this is a new residue, append the previous one

                        residue['type'] = aminoacid_abbr[previous_res_type]
                        residue['number'] = previous_res
                        residue['atoms'] = copy.deepcopy(atomlist)
                        residue['chain'] = current_res_chain

                        #check if the residue has CA, which means if the residue is completed for use.
                        has_CA = False
                        for atomCA in residue['atoms']:
                            if atomCA['type'] == 'CA':
                                has_CA = True
                                break
                                
                        if has_CA == True:        
                            residuelist.append(copy.deepcopy(residue))
                        else:
                            residue.clear()
                
                        atomlist.clear()
                        atomlist.append(copy.deepcopy(atom))
                        
                    else:
                        atomlist.append(copy.deepcopy(atom))

                    previous_res = current_res
                    previous_res_type = current_res_type

                check_res_number(residuelist)
                residue_chain_list.append(copy.deepcopy(residuelist))

        return residue_chain_list
    except Exception as e:
        print(e)
        return residue_chain_list

def match_seq_with_fasta(seq, fasta):
    
    i = 0 # seq pointer
    j = 0 # fasta pointer
    l = 0 # len of match part
    
    result = [] # list of (i, j, len)
    
    while i < len(seq) and j < len(fasta):
        if fasta[j] == seq[i]:
            while i + l < len(seq) and fasta[j + l] == seq[i + l] :
                l += 1
                
            result.append((i, j, l))
                
            j = j + l
            i = i + l
            
            l = 0

        else:
            j = j + 1
            
    if i != len(seq):
        return False
    else: 
        return result        

# get fasta sequence from fasta file, chains in chain list are all corresponding to fasta sequence 
# fasta_list is a list of fasta sequence, [,]
# chain_list is a list of chain list, each element in chain_list is a list of chains of corresponding fasta sequence [[a,b], [c,d]]

path1 = 'data/FASTA/2mole/'
path2 = 'data/FASTA/3mole/' 
mole2file = os.listdir(path1)
mole3file = os.listdir(path2)

for i, file in enumerate(mole2file):
    mole2file[i] = file.split('_')[0]

for i, file in enumerate(mole3file):
    mole3file[i] = file.split('_')[0]

def get_fasta_seq(pdb, directory):
    
    fasta_list = []
    chain_list = []

    if pdb in mole2file:
        with open(directory + pdb + '_1.fasta', 'r') as f1:
            lines = f1.readlines()
            fasta_1 = lines[1].strip()
            chain_1 = lines[0].split('|')[1]
            chain_1_list = chain_1.split(',')
            for i, chain in enumerate(chain_1_list):
                chain = chain.strip()
                if chain[-1] == ']':
                    chain_1_list[i] = chain[-2]
                else: 
                     chain_1_list[i] = chain[-1]
        with open(directory + pdb + '_2.fasta', 'r') as f2:
            lines = f2.readlines()
            fasta_2 = lines[1].strip()
            chain_2 = lines[0].split('|')[1]
            chain_2_list = chain_2.split(',')
            for i, chain in enumerate(chain_2_list):
                chain = chain.strip()
                if chain[-1] == ']':
                    chain_2_list[i] = chain[-2]
                else: 
                     chain_2_list[i] = chain[-1]
        fasta_list.append(fasta_1)
        fasta_list.append(fasta_2)
        chain_list.append(chain_1_list)
        chain_list.append(chain_2_list)
        return fasta_list, chain_list

    elif pdb in mole3file:
        with open(directory + pdb + '_1.fasta', 'r') as f1:
            lines = f1.readlines()
            fasta_1 = lines[1].strip()
            chain_1 = lines[0].split('|')[1]
            chain_1_list = chain_1.split(',')
            for i, chain in enumerate(chain_1_list):
                chain = chain.strip()
                if chain[-1] == ']':
                    chain_1_list[i] = chain[-2]
                else: 
                     chain_1_list[i] = chain[-1]
        with open(directory + pdb + '_2.fasta', 'r') as f2:
            lines = f2.readlines()
            fasta_2 = lines[1].strip()
            chain_2 = lines[0].split('|')[1]
            chain_2_list = chain_2.split(',')
            for i, chain in enumerate(chain_2_list):
                chain = chain.strip()
                if chain[-1] == ']':
                    chain_2_list[i] = chain[-2]
                else: 
                     chain_2_list[i] = chain[-1]

        with open(directory + pdb + '_3.fasta', 'r') as f3:
            lines = f3.readlines()
            fasta_3 = lines[1].strip()
            chain_3 = lines[0].split('|')[1]
            chain_3_list = chain_3.split(',')
            for i, chain in enumerate(chain_3_list):
                chain = chain.strip()
                if chain[-1] == ']':
                    chain_3_list[i] = chain[-2]
                else: 
                     chain_3_list[i] = chain[-1]
        fasta_list.append(fasta_1)
        fasta_list.append(fasta_2)
        fasta_list.append(fasta_3)
        chain_list.append(chain_1_list)
        chain_list.append(chain_2_list)
        chain_list.append(chain_3_list)
        return fasta_list, chain_list
    else:
        raise Exception('no fasta file')

# given the residue list and fasta sequence, match the residue number with the fasta sequence
# the residue number would change after this function

def match_seq_number_with_fasta(res_list, fasta_seq):
    seq_type_list = []
    for res in res_list:
        seq_type_list.append(res['type'])
        
    results = match_seq_with_fasta(seq_type_list, fasta_seq)
    if results == False:
        return False
    else:
        for result in results:
            for l in range(result[2]):
                res_list[result[0] + l]['number'] = result[1] + l
        return True   

# Given 2 parts of protein-protein complex, return the residue list of the complex
# Should be completed 2 residue lists

def get_interaction_residue_pair_new(dis_thred, reslistA, reslistB):
    res_pair = []
    proteinA = copy.deepcopy(reslistA)
    proteinB = copy.deepcopy(reslistB)
    
    for i, resA in enumerate(proteinA):     
        find_match_res_pair = 0
        ca1 = list(filter(lambda x: x['type'] == 'CA', resA['atoms']))[0]
        ca1_x = float(ca1['x'])
        ca1_y = float(ca1['y'])
        ca1_z = float(ca1['z'])
        res1_num = resA['number']
        
        for j, resB in enumerate(proteinB):
            find_match_res_pair = 0
            ca2 = list(filter(lambda x: x['type'] == 'CA', resB['atoms']))[0]
            ca2_x = float(ca2['x'])
            ca2_y = float(ca2['y'])
            ca2_z = float(ca2['z'])  
            res2_num = resB['number']
            if res1_num == res2_num:
                continue
        
            for atomA in resA['atoms']:
                Ax = float(atomA['x'])
                Ay = float(atomA['y'])
                Az = float(atomA['z'])        

                for atomB in resB['atoms']:
                    Bx = float(atomB['x'])
                    By = float(atomB['y'])
                    Bz = float(atomB['z'])

                    distance = math.sqrt((Ax-Bx)**2 + (Ay-By)**2 + (Az-Bz)**2)
          
                    if distance <= dis_thred:
     
                        c_alpha_dist = math.sqrt((ca1_x-ca2_x)**2 + (ca1_y-ca2_y)**2 + (ca1_z-ca2_z)**2)
                        #print(c_alpha_dist)

                        res_pair.append((resA, resB, c_alpha_dist))

                        find_match_res_pair = 1
                        break

                if find_match_res_pair == 1:
                    break

    return res_pair

def filter_data(pdb_chain_tuple, if_only_kd=False, if_2chain=False): 
    
    # do some filter here
    # filtered by Kd
    pdbfile_Kd = []
    with open('data/PPIdataindex_kd.txt', 'r') as f_y:
        lines = f_y.readlines()
        for line in lines:
            pdb = line.split()[0]
            pdbfile_Kd.append(pdb)

    # filter by 2chains
    if if_2chain == True and if_only_kd == True:
        pdblist = []
        chainlist = []
        for i, pdb in enumerate(pdb_chain_tuple[0]):
            if pdb in pdbfile_Kd and (len(pdb_chain_tuple[1][i][0]) == 1 and len(pdb_chain_tuple[1][i][1]) == 1):
                pdblist.append(pdb)
                chainlist.append(pdb_chain_tuple[1][i])

        return pdblist, chainlist

    elif if_2chain == False and if_only_kd == True:
        pdblist = []
        chainlist = []
        for i, pdb in enumerate(pdb_chain_tuple[0]):
            if pdb in pdbfile_Kd:
                pdblist.append(pdb)
                chainlist.append(pdb_chain_tuple[1][i])

        return pdblist, chainlist

    elif if_2chain == True and if_only_kd == False:
        pdblist = []
        chainlist = []
        for i, pdb in enumerate(pdb_chain_tuple[0]):
            if (len(pdb_chain_tuple[1][i][0]) == 1 and len(pdb_chain_tuple[1][i][1]) == 1):
                pdblist.append(pdb)
                chainlist.append(pdb_chain_tuple[1][i])

        return pdblist, chainlist

    elif if_2chain == False and if_only_kd == False:
        pdblist = []
        chainlist = []
        for i, pdb in enumerate(pdb_chain_tuple[0]):
            pdblist.append(pdb)
            chainlist.append(pdb_chain_tuple[1][i])

        return pdblist, chainlist

# concat a list of reslist into one reslist, 
# the number of the beginning residue should add the length of the fasta length of previous reslist 
# reslist is a list of reslist, [[], [], []]
# fastalist_index is a list of index of fasta sequence of each reslist, [index1, index2, index3]
# fasta_list is a list of fasta sequence, [fasta1, fasta2, fasta3]
# fasta_list and fastalist_index should be the same length, fasta_list[fastalist_index[i]] is the fasta sequence of reslist[i]

def concat_reslist(reslist, fastalist_index, fasta_list):

    # get the length of fasta sequence of each subreslist in reslist
    len_of_fasta_list = [] # previous_max
    number_of_res = len(reslist)
    for i in range(number_of_res):
        len_of_fasta = len(fasta_list[fastalist_index[i]])
        len_of_fasta_list.append(len_of_fasta)
    #print('len of each fasta:', len_of_fasta_list)
        
    
    number_list = [] # lists
    for subreslist in reslist:
        sub_number_list = []
        for res in subreslist:
            sub_number_list.append(res['number'])
        number_list.append(sub_number_list)
    #print('original number list:', number_list)

    cumulative_len = [0] + list(itertools.accumulate(len_of_fasta_list))[:-1]

    merged_list = []

    for sublist, cum_len in zip(number_list, cumulative_len):
        updated_sublist = [x + cum_len for x in sublist]
        merged_list.extend(updated_sublist)

    #print('number list after merge:', merged_list)

    merged_reslist = [item for sublist in reslist for item in sublist]
    for i, res in enumerate(merged_reslist):
        res['number'] = merged_list[i]

    return merged_reslist

# get the edge index of the graph from the interaction pairs
# only 2 parts in the inter_pairs
# len_pro1 is the length of the first part of the protein-protein complex

# note that here the pairs are from the same protein

def get_edge_index(inter_pairs):
    source_list = []
    des_list = []

    for pair in inter_pairs:
        source_res = int(pair[0]['number'])
        des_res = int(pair[1]['number'])
        if source_res != des_res:
            source_list.append(source_res)
            des_list.append(des_res)

    source = torch.unsqueeze(torch.tensor(source_list), 0)
    des = torch.unsqueeze(torch.tensor(des_list), 0)

    edge_index = torch.squeeze(torch.stack((source, des)))

    return edge_index


aminoacid_abbr = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PHE': 'F', 'TRP': 'W', 'TYR': 'Y', 'ASP': 'D', 'ASN': 'N', 'GLU': 'E', 'LYS': 'K', 'GLN': 'Q', 'MET': 'M', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'PRO': 'P', 'HIS': 'H', 'ARG': 'R', 'UNK': 'X'}
thred = 3.5
pdb_chain_tuple = get_chainlist_from_indexfile('data/index_example.txt')
pdblist, chainlist = filter_data(pdb_chain_tuple, if_only_kd=False, if_2chain=False)

for pdb, prot_pair in zip(pdblist, chainlist):
    print(pdb)
    if len(prot_pair) != 2:
        print('no 2 pro-pro pair!!')
        continue
    flag = 1
    
    residuelistA = get_residue_list_from_file('data/pdbqt/' + pdb + '_atom_processed' + '.pdbqt', prot_pair[0])
    residuelistB = get_residue_list_from_file('data/pdbqt/' + pdb + '_atom_processed' + '.pdbqt', prot_pair[1])
    if len(residuelistA) == 0 or len(residuelistB) == 0:
        print('no residue list')
        continue

    try:
        fasta_list, chain_list = get_fasta_seq(str.upper(pdb), 'data/FASTA/mixed/')
    except Exception as e:
        print(e)
        continue

    chain_fastalist_indexA = []
    chain_len_listA = []
    fastaA = []

    try:
        for reslist in residuelistA: # for each chain in the residuelistA:
            chain_name = reslist[0]['chain'] # get the chain name

            # find the fasta sequence corresponding to this chain
            for i, chains in enumerate(chain_list):
                if chain_name in chains:
                    fasta_seq = fasta_list[i]
                    chain_len_listA.append(len(fasta_seq))
                    chain_fastalist_indexA.append(i)
                    fastaA.append(fasta_seq)
                    
                    a = match_seq_number_with_fasta(reslist, fasta_seq)
                    if a == False:
                        print(pdb + ' fasta1 cannot match')
                        flag = 0
                    break
        if flag == 0:
            continue

        concat_reslistA = concat_reslist(residuelistA, chain_fastalist_indexA, fasta_list)
        total_fastaA_len = sum(len(s) for s in fastaA)

        chain_fastalist_indexB = []
        chain_len_listB = []
        fastaB = []

        for reslist in residuelistB: # for each chain in the residuelistA:
            chain_name = reslist[0]['chain'] # get the chain name

            # find the fasta sequence corresponding to this chain
            for i, chains in enumerate(chain_list):
                if chain_name in chains:
                    fasta_seq = fasta_list[i]
                    chain_len_listB.append(len(fasta_seq))
                    chain_fastalist_indexB.append(i)
                    fastaB.append(fasta_seq)
                    
                    a = match_seq_number_with_fasta(reslist, fasta_seq)
                    if a == False:
                        print(pdb + ' fasta2 cannot match')
                        flag = 0
                    break
        if flag == 0:
            continue

        concat_reslistB = concat_reslist(residuelistB, chain_fastalist_indexB, fasta_list)
        intra_pairsA = get_interaction_residue_pair_new(thred, concat_reslistA, concat_reslistA)
        intra_pairsB = get_interaction_residue_pair_new(thred, concat_reslistB, concat_reslistB)

        edge_index1 = get_edge_index(intra_pairsA)
        edge_index2 = get_edge_index(intra_pairsB)

        info_save1 = (intra_pairsA, edge_index1, fastaA)
        info_save2 = (intra_pairsB, edge_index2, fastaB)
                
        filename1 = 'data/graph_construct/individual_graph/' + pdb + '_1'
        f1 = open(filename1, 'wb') 
        pickle.dump(info_save1, f1)
        f1.close()

        filename2 = 'data/graph_construct/individual_graph/' + pdb + '_2'
        f2 = open(filename2, 'wb')
        pickle.dump(info_save2, f2)
        f2.close()
    except Exception as e:
        print(e)
        continue