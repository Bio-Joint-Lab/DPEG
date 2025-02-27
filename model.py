import torch_geometric
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
class SelectiveStateSpaceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelectiveStateSpaceModel, self).__init__()

        self.gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.attention_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention_score = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, sequence_length, input_dim = x.size()
        hidden_state = torch.zeros(batch_size, self.gru_cell.hidden_size).to(x.device)
        outputs = []
        for t in range(sequence_length):
            current_input = x[:, t, :]
            combined = torch.cat([current_input, hidden_state], dim=-1)
            gate_output = torch.sigmoid(self.gate(combined))
            hidden_state = self.gru_cell(current_input, hidden_state)
            hidden_state = gate_output * hidden_state
            attention_input = torch.cat([hidden_state, hidden_state], dim=-1)
            attention_hidden = torch.tanh(self.attention_layer(attention_input))
            attention_weight = torch.softmax(self.attention_score(attention_hidden), dim=1)

            hidden_state = hidden_state * attention_weight
            outputs.append(hidden_state)
        outputs = torch.stack(outputs, dim=1)
        return outputs[:, -1, :]

class ProteinGraphModule(nn.Module):
    def __init__(self, num_features_pro=33, output_dim=128, fc_output_dim=1024,dropout=0.2):
        super(ProteinGraphModule, self).__init__()

        self.ssm = SelectiveStateSpaceModel(33, 66)


        self.conv1 = GATv2Conv(num_features_pro*2, num_features_pro*2, heads=2)
        self.conv2 = GATv2Conv(num_features_pro*4, num_features_pro * 4, heads=1)
        self.conv3 = GATv2Conv(num_features_pro*4, num_features_pro * 4, heads=1)
        self.pro_fc1 = nn.Linear(num_features_pro * 4, fc_output_dim)
        self.pro_fc2 = nn.Linear(fc_output_dim, output_dim)


        self.fc1 = nn.Linear(2*output_dim, fc_output_dim)
        self.fc2 = nn.Linear(fc_output_dim, 512)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):

        x = self.ssm(x)

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x,edge_index)

        x = torch_geometric.nn.global_mean_pool(x, batch)

        x = self.relu(self.pro_fc1(x))
        x = self.dropout(x)
        x = self.pro_fc2(x)
        x = self.dropout(x)

        return x
class ProteinActivityPredictionModel(nn.Module):
    def __init__(self, num_features_pro=33, output_dim=128, fc_output_dim=1024, n_output=1, dropout=0.2):
        super(ProteinActivityPredictionModel, self).__init__()
        print('ProteinActivityPredictionModel Loading ...')

        self.protein1_module = ProteinGraphModule(num_features_pro, output_dim, fc_output_dim)
        self.protein2_module = ProteinGraphModule(num_features_pro, output_dim, fc_output_dim)

        self.final_fc1 = nn.Linear(2 * output_dim, fc_output_dim)
        self.final_fc2 = nn.Linear(fc_output_dim, 512)
        self.out = nn.Linear(512, n_output)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.sigmoid = nn.Sigmoid()


    def forward(self, data1, data2):

        x1 = self.protein1_module(data1.x, data1.edge_index, data1.batch)
        x2 = self.protein2_module(data2.x, data2.edge_index, data2.batch)

        xc = torch.cat([x1, x2], dim=-1)
        xc = self.final_fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.final_fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.sigmoid(self.out(xc))

        return out
