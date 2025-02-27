#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Rick Pang
# email:pangshunpeng@gmail.com
# datetime:2024/11/26 20:58
# software: PyCharm
import pandas as pd
import json
import os
def get_list(seq_file):
    pd_file = pd.read_csv(seq_file,sep='\t', header=None)
    names = pd_file[0]
    seqs = pd_file[1]
    # print(type(names))
    large_names =[]
    large_seqs = []
    count=0
    for name,seq in  zip(names,seqs):
        if (len(seq)>1500):
            # print(f'name:{name}:{len(seq)}')
            count = count+1
            large_names.append(name)
    return large_names
def check_cases(seq_file,pair_file):
    pd_file = pd.read_csv(pair_file, sep='\t', header=None)
    protein_1s = pd_file[0]
    protein_2s = pd_file[1]
    large_names = get_list(seq_file)
    count = 0
    print(len(large_names))
    for protein_1, protein_2 in zip(protein_1s, protein_2s):
        if protein_1 in large_names or protein_2 in large_names:
            count = count+1
    print(count)
    print(pd_file.shape[0]-count)
def read_proteins(seq_file,length):
    pd_file = pd.read_csv(seq_file, sep='\t', header=None)
    # print(pd_file.shape)
    filtered_df = pd_file[pd_file[1].str.len() <= length].reset_index(drop=True)
    names = filtered_df[0]
    seqs = filtered_df[1]
    # print(filtered_df.shape)
    # print(names)
    # print(seqs)
    return filtered_df
def read_pairs(seq_file,pair_file,length):
    # read sequence file
    seq_df = pd.read_csv(seq_file, sep='\t', header=None)  # 假设序列文件以制表符分隔
    filtered_seq_df = seq_df[seq_df[1].str.len() <= length].reset_index(drop=True)

    # larger than 2000
    long_sequences = seq_df[seq_df[1].str.len() > length][0].tolist()
    num_removed = len(long_sequences)
    print("removed protein:", num_removed)
    # 读取配对文件
    pd_file = pd.read_csv(pair_file, sep='\t', header=None)
    # 统计正负样本个数
    counts = pd_file[2].value_counts()
    # 原始配对文件样本数量
    print("original shape:",pd_file.shape)
    print('positive:',counts.get(1,0))
    print('negative:',counts.get(0,0))
    filtered_action_df = pd_file[
        ~pd_file[0].isin(long_sequences) &
        ~pd_file[1].isin(long_sequences)
        ].reset_index(drop=True)
    protein1 = filtered_action_df[0]
    protein2 = filtered_action_df[1]
    proteins = list(set(list(protein1) + list(protein2)))
    # 删选后的样本数量
    print("filtered shape:",filtered_action_df.shape)
    # 统计筛选后正负样本个数
    filter_counts = filtered_action_df[2].value_counts()
    print('positive:', filter_counts.get(1, 0))
    print('negative:', filter_counts.get(0, 0))
    # 删除的样本数量。
    print('pairs removed:',pd_file.shape[0]-filtered_action_df.shape[0])
    print(f'number of proteins:{len(proteins)}')
    print("--"*20)
    return filtered_action_df,filtered_seq_df
def get_subdata_path():
    file_path = 'data_file_path.json'

    # 打开并读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
if __name__ == '__main__':

    PPIdataset = ['BioGRID_S', 'BioGRID_H', 'multiple_species_01', 'multiple_species_10', 'multiple_species_25',
         'multiple_species_40', 'multiple_species_full', 'DeepFE-PPI_core', 'PIPR_core', 'origin_PIPR']
    subdata_path = get_subdata_path()
    protein_data_path = os.path.join('data', 'benchmarks')
    for i in [2,3,4,5,6]:
        dataset = PPIdataset[i]

        pair_file_path = os.path.join(protein_data_path, subdata_path[dataset]['pair'])
        seq_file_path = os.path.join(protein_data_path, subdata_path[dataset]['database'])

        root_path = os.path.dirname(pair_file_path)

        action_file_name = os.path.basename(pair_file_path)
        action_name_prefix = os.path.splitext(action_file_name)[0]

        seq_file_name = os.path.basename(seq_file_path)
        seq_name_prefix = os.path.splitext(seq_file_name)[0]

        output_pair = os.path.join(root_path, action_name_prefix+'.l1500.tsv')
        output_dic =os.path.join(root_path, seq_name_prefix+'.l1500.tsv')

        print(seq_file_path)
        print(pair_file_path)
        # check_cases(seq_file_path, pair_file_path)

        act, dic = read_pairs(seq_file_path,pair_file_path,1500)
        # act.to_csv(output_pair, sep='\t', index=False,header=False)
        # dic.to_csv(output_dic, sep='\t', index=False,header=False)






