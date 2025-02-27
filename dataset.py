import os
import random

import numpy as np
import pickle
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
from data_process import process_proteins, read_ppi_pairs, get_proteins_contacts


def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']
pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_weight_table['X'] = np.average([res_weight_table[k] for k in res_weight_table.keys()])

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pka_table['X'] = np.average([res_pka_table[k] for k in res_pka_table.keys()])

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkb_table['X'] = np.average([res_pkb_table[k] for k in res_pkb_table.keys()])

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pkx_table['X'] = np.average([res_pkx_table[k] for k in res_pkx_table.keys()])

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_pl_table['X'] = np.average([res_pl_table[k] for k in res_pl_table.keys()])

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph2_table['X'] = np.average([res_hydrophobic_ph2_table[k] for k in res_hydrophobic_ph2_table.keys()])

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}
res_hydrophobic_ph7_table['X'] = np.average([res_hydrophobic_ph7_table[k] for k in res_hydrophobic_ph7_table.keys()])

# nomarlize the residue feature
res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)

# one ont encoding
def one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


# one ont encoding with unknown symbol
def one_hot_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(one_hot_encoding_unk(atom.GetSymbol(),
                                         ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                          'Pt', 'Hg', 'Pb', 'X']) +
                    one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def seq_feature(seq):
    residue_feature = []
    for residue in seq:
        # replace some rare residue with 'X'
        if residue not in pro_res_table:
            residue = 'X'
        res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                         1 if residue in pro_res_polar_neutral_table else 0,
                         1 if residue in pro_res_acidic_charged_table else 0,
                         1 if residue in pro_res_basic_charged_table else 0]
        res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue],
                         res_pkx_table[residue],
                         res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
        residue_feature.append(res_property1 + res_property2)

    pro_hot = np.zeros((len(seq), len(pro_res_table)))
    pro_property = np.zeros((len(seq), 12))
    for i in range(len(seq)):
        pro_hot[i,] = one_hot_encoding_unk(seq[i], pro_res_table)
        pro_property[i,] = residue_feature[i]

    seq_feature = np.concatenate((pro_hot, pro_property), axis=1)
    return seq_feature

# data write to csv file
def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')

def create_PPI_dataset(dataset='BioGRID_S', data_process_path='/home/pang/datasets/process/benchmarks'):
    print("creating ppi data for ", dataset)
    data_process_path_temp = os.path.join(data_process_path, dataset)
    seq_save_file = os.path.join(data_process_path_temp, "seq.pkl")
    process_proteins(dataset)
    protein_1, protein_2, interactions = read_ppi_pairs(dataset)
    data = []
    for p1, p2, a in zip(protein_1, protein_2, interactions):
        data.append([p1, p2, a])

    proteins = list(set(list(protein_1) + list(protein_2)))
    positive_sum = np.sum(np.array(interactions))
    negative_sum = len(interactions) - positive_sum
    print("the number of proteins:", len(proteins))
    print("the number of ppi entries:", len(interactions))
    print("the number of positive ppi entries:", positive_sum)
    print("the number of negative ppi entries:", negative_sum)
    dataset_temp = dataset
    if dataset == 'multiple_species_01' or dataset == 'multiple_species_10' or dataset == 'multiple_species_25' or dataset == 'multiple_species_40' or dataset == 'multiple_species_full':
        dataset_temp = 'multiple_species'
        seq_save_file = os.path.join(data_process_path, dataset_temp, "seq.pkl")
    contacts = get_proteins_contacts(dataset_temp)

    with open(seq_save_file, "rb") as f:
        keys, seqs = pickle.load(f)
    seq_data = {}
    for key, seq in zip(keys, seqs):
        seq_data[key] = seq

    target_graphs = {}
    print("creating portein graphs ...")
    for i in tqdm(range(len(proteins))):
        key = proteins[i]
        seq = seq_data[key]
        target_x = seq_feature(seq)
        contact = contacts[key]
        contact = contact + np.identity(len(seq))
        index_row, index_col = np.where(contact >= 0.5)
        target_edge_index = []
        for i, j in zip(index_row, index_col):
            target_edge_index.append([i, j])
        target_graph = (target_x, target_edge_index)
        target_graphs[key] = target_graph

    train_dataset = PPIDataset(dataset=dataset, data=data, target_graphs=target_graphs)
    return train_dataset

class PPIDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='BioGRID_S',
                 data=None, seq=None, target_graphs=None, transform=None, pre_transform=None):
        super(PPIDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.target_graphs = target_graphs
        self.seq = seq
        self.data = data
        self.process(self.data, self.target_graphs, self.seq)

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '_data_mol.pt', self.dataset + '_data_pro.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, data, target_graphs, seq):
        data_list_pro_1 = []
        data_list_pro_2 = []
        data_len = len(data)
        print('loading tensors ...')
        print("initing dataset ...")
        for i in tqdm(range(data_len)):
            data_entry = data[i]
            pro1 = data_entry[0]
            pro2 = data_entry[1]
            y = data_entry[2]
            pro1_x, pro1_edge_index = target_graphs[pro1]
            pro2_x, pro2_edge_index = target_graphs[pro2]

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GNNData_pro1 = DATA.Data(x=torch.Tensor(pro1_x),
                                     edge_index=torch.LongTensor(pro1_edge_index).transpose(1, 0),
                                     y=torch.FloatTensor([y]))

            GNNData_pro2 = DATA.Data(x=torch.Tensor(pro2_x),
                                     edge_index=torch.LongTensor(pro2_edge_index).transpose(1, 0),
                                     y=torch.FloatTensor([y]))
            data_list_pro_1.append(GNNData_pro1)
            data_list_pro_2.append(GNNData_pro2)

        if self.pre_filter is not None:
            data_list_pro_1 = [data for data in data_list_pro_1 if self.pre_filter(data)]
            data_list_pro_2 = [data for data in data_list_pro_2 if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list_pro_1 = [self.pre_transform(data) for data in data_list_pro_1]
            data_list_pro_2 = [self.pre_transform(data) for data in data_list_pro_2]
        self.data_list_pro_1 = data_list_pro_1
        self.data_list_pro_2 = data_list_pro_2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data_list_pro_1[idx], self.data_list_pro_2[idx]

if __name__ == '__main__':
    create_PPI_dataset(dataset='BioGRID_S')
