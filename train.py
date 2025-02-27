import multiprocessing
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score, matthews_corrcoef,roc_curve,confusion_matrix
# sys.path.append('./')
# from metric import *
from model import ProteinActivityPredictionModel
from dataset import create_PPI_dataset
from utils import train, predicting, collate,logger,EarlyStopping,set_cuda_name

def count_positives_negatives(data_labels):
    data, labels = data_labels
    positives = (labels == 1).sum().item()
    negatives = (labels == 0).sum().item()
    return positives, negatives

PPIdataset = \
['BioGRID_S', 'BioGRID_H', 'multiple_species_01', 'multiple_species_10', 'multiple_species_25', 'multiple_species_40',
 'multiple_species_full', 'DeepFE-PPI_core', 'virus-human'][int(sys.argv[1])]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]

set_cuda_name(cuda_name)
logger = logger()

print('cuda_name:', cuda_name)
print('dataset:', PPIdataset)

model_type = ProteinActivityPredictionModel
TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

models_dir = 'models'
results_dir = 'results'
if not os.path.exists(os.path.join(results_dir, PPIdataset)):
    os.makedirs(os.path.join(results_dir, PPIdataset))

if not os.path.exists(os.path.join(models_dir, PPIdataset)):
    os.makedirs(os.path.join(models_dir, PPIdataset))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')

dataset = create_PPI_dataset(PPIdataset)
labels = torch.tensor([dataset[i][0].y.item() for i in range(len(dataset))])
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(dataset,labels)):
    early_stopping = EarlyStopping(patience=25,verbose=True)
    model = model_type()
    model.to(device)
    model_st = model_type.__name__

    best_model_file_name = None
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    train_indices = torch.tensor(train_idx)
    test_indices = torch.tensor(test_idx)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    train_positives, train_negatives = count_positives_negatives((train_dataset, labels[train_idx]))
    test_positives, test_negatives = count_positives_negatives((test_dataset, labels[test_idx]))

    print(f"Fold {fold}:")
    print(f"\tTrain data size: {len(train_dataset)}")
    print(f"\t\tPositive samples: {train_positives}")
    print(f"\t\tNegative samples: {train_negatives}")
    print(f"\tTest data size: {len(test_dataset)}")
    print(f"\t\tPositive samples: {test_positives}")
    print(f"\t\tNegative samples: {test_negatives}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4,
                                               shuffle=True,
                                               collate_fn=collate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,

                                              collate_fn=collate)

    model_file_name = os.path.join(models_dir, PPIdataset,f'model_{model_type.__name__}_{PPIdataset}_fold_{fold}.model')


    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1, loss_fn, TRAIN_BATCH_SIZE)

        if (epoch + 1) % 5 == 0:
            print('验证模型性能...')
            T, P = predicting(model, device, test_loader)
            test_f1 = f1_score(T, np.where(P >= 0.5, 1, 0))
            test_accuracy = accuracy_score(T, np.where(P >= 0.5, 1, 0))
            test_recall = recall_score(T, np.where(P >= 0.5, 1, 0))
            test_precision = precision_score(T, np.where(P >= 0.5, 1, 0))
            test_MCC = matthews_corrcoef(T, np.where(P >= 0.5, 1, 0))
            metrics = 0.85 * test_accuracy + 0.25 * test_precision + 0.85 * test_f1 + 0.25 * test_recall

            print(f"Epoch {epoch + 1}: Test Accuracy: {test_accuracy:.4f}, F1 Score: {test_f1:.4f}\n,"
                  f" Precision:{test_precision}, MCC:{test_MCC},recall:{test_recall}")

            if early_stopping(metrics):
                print(f"Precision improved, saving model at epoch {epoch + 1}")
                torch.save(model.state_dict(), model_file_name)

            if early_stopping.early_stop:
                print("Precision did not improve, stopping early.")
                break

    print('测试模型...')
    print('all trainings done. Testing...')
    test_positives, test_negatives = count_positives_negatives((test_dataset, labels[test_idx]))
    print(f"\tTest data size: {len(test_dataset)}")
    print(f"\t\tPositive samples: {test_positives}")
    print(f"\t\tNegative samples: {test_negatives}")

    model_p = model_type().to(device)
    model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
    test_T, test_P = predicting(model_p, device, test_loader)

    predicted_labels = (test_P > 0.5).astype(np.float32)


    TP = np.sum((predicted_labels == 1) & (test_T == 1))
    TN = np.sum((predicted_labels == 0) & (test_T == 0))
    FP = np.sum((predicted_labels == 1) & (test_T == 0))
    FN = np.sum((predicted_labels == 0) & (test_T == 1))

    test_auc = roc_auc_score(test_T, test_P)
    test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_f1_score = f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_accuracy = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
    test_MCC = matthews_corrcoef(test_T, np.where(test_P >= 0.5, 1, 0))
    test_specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    conf_mat = confusion_matrix(test_T, np.where(test_P >= 0.5, 1, 0))

    result_str = (
        f"test result:\n"
        f"test_accuracy: {test_accuracy}\n"
        f"test_precision: {test_precision}\n"
        f"test_recall: {test_recall}\n"
        f"test_specificity: {test_specificity}\n"
        f"test_MCC: {test_MCC}\n"
        f"test_f1_score: {test_f1_score}\n"
        f"test_auc: {test_auc}\n"
    )

    print(result_str)
    save_file = os.path.join(results_dir, PPIdataset, f'test_restult_fold_{fold}_{model_type.__name__}.txt')
    open(save_file, 'w').writelines(result_str)

    size_str = (
        f"size result:\n"
        f"Train data size: {len(train_dataset)}\n"
        f"Positive samples: {train_positives}\n"
        f"Negative samples: {train_negatives}\n"
        f"Test data size: {len(test_dataset)}\n"
        f"Positive samples: {test_positives}\n"
        f"Negative samples: {test_negatives}\n"
        f"confusion matrix: {conf_mat}\n"
    )

    size_save_file = os.path.join(results_dir, PPIdataset, f'size_statistic_fold_{fold}_.txt')
    open(size_save_file, 'w').writelines(size_str)

    logger.info("all trainings done. Testing...")
    logger.info(save_file)
    logger.info(result_str)
    logger.info(size_save_file)
    logger.info(size_str)