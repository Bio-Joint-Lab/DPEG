from torch_geometric.data import InMemoryDataset, DataLoader, Batch
import torch
import logging
import os
from datetime import datetime

cuda_name=None

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, TRAIN_BATCH_SIZE):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 10
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        optimizer.zero_grad()
        output = model(data_mol, data_pro)
        loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * TRAIN_BATCH_SIZE,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output = model(data_mol, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


# prepare the protein and drug pairs
def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    return batchA, batchB


def init_model(model):
    for param in model.parameters():
        param.data.normal_(0.0, 0.01)

def logger(log_file=os.path.join('./logger', datetime.now().strftime("%Y%m%d") + ".logger")):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.0):
        """
        Args:
            patience (int): 如果 precision 在 patience 个 epoch 内没有提升，则停止训练
            verbose (bool): 是否打印详细信息
            delta (float): 精度提升的最小变化，用来判定是否提升
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_precision = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_precision):
        # 初始化最佳 precision
        if self.best_precision is None:
            self.best_precision = val_precision
            return True  # 保存模型
        # 如果 precision 没有提升
        elif val_precision < self.best_precision + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False  # 不保存模型
        # 如果 precision 提升，重置计数器
        else:
            self.best_precision = val_precision
            self.counter = 0
            return True  # 保存模型

def set_cuda_name(value):
    global cuda_name
    cuda_name = value

def get_cuda_name():
    global cuda_name
    return cuda_name