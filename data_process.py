import os
import pandas as pd
import numpy as np
import torch
import esm
import traceback
import pickle
from tqdm import tqdm
from utils import logger, get_cuda_name
import json

logger = logger()

def get_subdata_path():
    file_path = 'data_file_path.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_proteins(seq_file):
    pd_file = pd.read_csv(seq_file, sep='\t', header=None)
    names = pd_file[0]
    seqs = pd_file[1]
    return names, seqs

def read_pairs(pair_file):
    pd_file = pd.read_csv(pair_file, sep='\t', header=None)
    protein_1 = pd_file[0]
    protein_2 = pd_file[1]
    interactions = pd_file[2]
    return protein_1, protein_2, interactions

def contact_predict(names, seqs, save_path):
    assert len(names) == len(seqs), 'protein names should have the same length as protein seqs.'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cuda_name = get_cuda_name()
    print(f"pretrained model is used:{cuda_name}")
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    torch.hub.set_dir('./pretrained')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()

    model.eval()
    model.to(device)
    contacts = {}
    with torch.no_grad():
        for i in tqdm(range(len(names))):
            name, seq = names[i], seqs[i]
            save_file = os.path.join(save_path, name + '_contact.npy')
            try:
                if os.path.exists(save_file):
                    contacts[name] = np.load(save_file)
                    continue
                if len(seq) <= 800:
                    _, _, batch_tokens = batch_converter([(name, seq)])
                    batch_tokens = batch_tokens.to(device)
                    contact = model.predict_contacts(batch_tokens)
                    contacts[name] = contact.to("cpu").numpy()
                    logger.info(len(seq))
                    logger.info(contacts[name])
                elif len(seq) > 800 and len(seq) <= 1000:
                    _, _, batch_tokens = batch_converter([(name, seq)])
                    model.to("cpu")
                    contact = model.predict_contacts(batch_tokens)
                    contacts[name] = contact.numpy()
                    model = model.to(device)
                    logger.info(len(seq))
                    logger.info(contacts[name].shape)

                elif len(seq)>1000:
                    cut_L = 2500
                    step = 2000
                    count = L = len(seq)
                    contact = np.zeros((1, L, L))
                    start = 0
                    # print(L)
                    model.to("cpu")
                    while (count > 0):
                        temp_L = min(cut_L, count)
                        temp_sub_seq = seq[start: start + temp_L]
                        _, _, batch_tokens_temp = batch_converter([(name + "_" + str(i), temp_sub_seq)])
                        contact_temp = model.predict_contacts(batch_tokens_temp)

                        contact[:, start:start + temp_L, start:start + temp_L] = (contact[:, start:start + temp_L,
                                                                                  start:start + temp_L] + contact_temp.numpy()) / 2.0

                        start = start + step
                        count = count - step
                    contacts[name] = contact

                    print(contacts[name].shape, contact_temp.shape, len(seq))

                    logger.info(len(seq))
                    logger.info(contacts[name].shape)
                    logger.info(contact_temp.shape)

                    model.to(device)

                np.save(save_file, contacts[name])

            except:
                traceback.print_exc()
                print('peocess error, protein information:')
                print(save_file)
                print(name)
                print(seq)
                exit(0)
                pass


def read_ppi_pairs(dataset='BioGRID_S'):
    protein_data_path = os.path.join('data', 'benchmarks')
    subdata_path = get_subdata_path()
    print(f"process {dataset} pairs ......")
    protein_pair_path = os.path.join(protein_data_path,subdata_path[dataset]['pair'])
    protein_1, protein_2, interactions = read_pairs(protein_pair_path)
    return protein_1, protein_2, interactions


def process_proteins(dataset='BioGRID_S',
                     data_process_path='./process/benchmarks'):
    data_process_path_temp = os.path.join(data_process_path, dataset)
    protein_data_path = os.path.join('data', 'benchmarks')

    subdata_path = get_subdata_path()
    print(f"process {dataset} dataset ......")
    protein_data_path = os.path.join(protein_data_path, subdata_path[dataset]['database'])

    if dataset == 'multiple_species_01' or dataset == 'multiple_species_10' or dataset == 'multiple_species_25' or dataset == 'multiple_species_40' or dataset == 'multiple_species_full':
        data_process_path_temp = os.path.join(data_process_path, 'multiple_species')

    contact_path = os.path.join(data_process_path_temp, "contacts")
    seq_save_file = os.path.join(data_process_path_temp, "seq.pkl")
    names, seqs = read_proteins(protein_data_path)
    contact_predict(names, seqs, contact_path)
    with open(seq_save_file, "wb") as f:
        pickle.dump([names, seqs], f)


def get_proteins_contacts(dataset='BioGRID_S',
                          data_process_path='./process/benchmarks'):
    data_process_path = os.path.join(data_process_path, dataset)
    contact_path = os.path.join(data_process_path, "contacts")
    print("read contacts ...")
    contacts = {}
    contact_files = os.listdir(contact_path)
    for i in tqdm(range(len(contact_files))):
        file = contact_files[i]
        protein_name = file[:-12]
        contact = np.load(os.path.join(contact_path, file))
        contacts[protein_name] = contact.squeeze(0)
    return contacts


if __name__ == '__main__':
    from utils import set_cuda_name
    set_cuda_name('cuda:1')
    dataset = ['BioGRID_S']
    for set in dataset:
        process_proteins(set)
        read_ppi_pairs(set)
        contacts = get_proteins_contacts(set)
        print(len(contacts))
