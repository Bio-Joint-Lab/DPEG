#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Rick Pang
# email:pangshunpeng@gmail.com
# datetime:2025/6/18 9:49
# software: PyCharm
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Rick Pang
# email:pangshunpeng@gmail.com
# datetime:2024/11/20 15:12
# software: PyCharm
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve,roc_curve,auc,average_precision_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, \
    accuracy_score, matthews_corrcoef,roc_curve,confusion_matrix
from dataset import create_PPI_dataset
from utils import predicting, collate, logger, set_cuda_name
from model import ProteinActivityPredictionModel

logger = logger()

# 检查命令行参数
if len(sys.argv) != 4:
    print("使用方法: python indepTest.py <testDataset> <model_file_name> <cuda_name>")
    print("示例: python indepTest.py virus-human /path/to/model.model cuda:0")
    sys.exit(1)

# 从命令行参数获取输入
testDataset = sys.argv[1]
model_file_name = sys.argv[2]
cuda_name = sys.argv[3]

print(f'测试数据集: {testDataset}')
print(f'模型文件: {model_file_name}')
print(f'CUDA设备: {cuda_name}')

if not os.path.exists(model_file_name):
    print(f"错误: 模型文件 '{model_file_name}' 不存在")
    sys.exit(1)

set_cuda_name(cuda_name)

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
# WED = 1e-4
NUM_EPOCHS = 1000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
logger.info('epoch: {}, learning rate: {}, train_batch: {}, test_batch: {}'.format(NUM_EPOCHS, LR, TRAIN_BATCH_SIZE,
                                                                                   TEST_BATCH_SIZE))
models_dir = 'models'
results_dir = 'results'

if not os.path.exists(os.path.join(results_dir, testDataset)):
    os.makedirs(os.path.join(results_dir, testDataset))

result_str = ''
USE_CUDA = torch.cuda.is_available()
device = torch.device(cuda_name if USE_CUDA else 'cpu')

dataset = create_PPI_dataset(testDataset)

model = ProteinActivityPredictionModel()
model.to(device)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

data_loader = torch.utils.data.DataLoader(dataset, batch_size=TEST_BATCH_SIZE, shuffle=False,
                                          collate_fn=collate)


print(model_file_name)
model_p = ProteinActivityPredictionModel().to(device)
model_p.load_state_dict(torch.load(model_file_name, map_location=cuda_name))
test_T, test_P = predicting(model_p, device, data_loader)

test_f1 = f1_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_accuracy = accuracy_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_recall = recall_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_precision = precision_score(test_T, np.where(test_P >= 0.5, 1, 0))
test_MCC = matthews_corrcoef(test_T, np.where(test_P >= 0.5, 1, 0))
test_auc = roc_auc_score(test_T, test_P)

result_str = (
    f"test result:\n"
    f"test_accuracy: {test_accuracy}\n"
    f"test_precision: {test_precision}\n"
    f"test_recall: {test_recall}\n"
    f"test_MCC: {test_MCC}\n"
    f"test_auc: {test_auc}\n"
    f""f"test_f1_score: {test_f1}\n"
)

print(result_str)
save_file = os.path.join(results_dir, testDataset, f'test_restult.txt')
open(save_file, 'w').writelines(result_str)