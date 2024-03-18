import torch
from torch_geometric.nn.models import AttentiveFP
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import L1Loss
import torch.nn.functional as F
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
from itertools import chain
import copy
import numpy as np
import matplotlib.pyplot as plt
import os

if_edge_norm = False
remove_isolated = False

datalist_inter = []
datalist_intra1 = []
datalist_intra2 = []

# get test set file
path = 'data/graph/inter_graph/'
filenames = os.listdir(path)
filenames.sort()
print(filenames)

for filename in filenames:
    pdb = filename
    f_path1 = 'data/graph/inter_graph/' + pdb
    f_path2 = 'data/graph/individual_graph/' + pdb + '_1'
    f_path3 = 'data/graph/individual_graph/' + pdb + '_2'

    print(pdb)
    try:
        with open(f_path1, 'rb') as f_graph1:
            graph = pickle.load(f_graph1)
            graph.edge_attr = graph.edge_attr.float()
            y = graph.y
            datalist_inter.append(graph)
            
        with open(f_path2, 'rb') as f_graph2:
            graph = pickle.load(f_graph2)
            graph.edge_attr = graph.edge_attr.float()
            datalist_intra1.append(graph)

        with open(f_path3, 'rb') as f_graph3:
            graph = pickle.load(f_graph3)
            graph.edge_attr = graph.edge_attr.float()
            datalist_intra2.append(graph)

    except Exception as e:
        print(e)
        continue

test_loader_inter = DataLoader(datalist_inter, batch_size=16)
test_loader_intra1 = DataLoader(datalist_intra1, batch_size=16)
test_loader_intra2 = DataLoader(datalist_intra2, batch_size=16)


# Define the model
class AttentiveFPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout):
        super(AttentiveFPModel, self).__init__()
        self.model = AttentiveFP(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.model(x, edge_index, edge_attr, batch)

class GraphNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2):
        super(GraphNetwork, self).__init__()
        self.graph1 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph2 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph3 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        
        self.fc1 = torch.nn.Linear(out_channels * 3, linear_out1)
        self.fc2 = torch.nn.Linear(linear_out1, linear_out2)

    def forward(self, inter_data, intra_data1, intra_data2):

        inter_graph = self.graph1(inter_data)
        intra_graph1 = self.graph2(intra_data1)
        intra_graph2 = self.graph3(intra_data2)

        x = torch.cat([inter_graph, intra_graph1, intra_graph2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentiveFPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout):
        super(AttentiveFPModel, self).__init__()
        self.model = AttentiveFP(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        return self.model(x, edge_index, edge_attr, batch)

class GraphNetwork(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2):
        super(GraphNetwork, self).__init__()
        self.graph1 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph2 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        self.graph3 = AttentiveFPModel(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout)
        
        self.fc1 = torch.nn.Linear(out_channels * 3, linear_out1)
        self.fc2 = torch.nn.Linear(linear_out1, linear_out2)

    def forward(self, inter_data, intra_data1, intra_data2):
        # x1, edge_index1, edge_attr1, batch1 = inter_data.x, inter_data.edge_index, inter_data.edge_attr, inter_data.batch
        inter_graph = self.graph1(inter_data)
        # print(inter_graph.shape)
        intra_graph1 = self.graph2(intra_data1)
        # print(intra_graph1.shape)
        intra_graph2 = self.graph3(intra_data2)
        # print(intra_graph2.shape)

        x = torch.cat([inter_graph, intra_graph1, intra_graph2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, data_loader_inter, data_loader_intra1, data_loader_intra2, optimizer, criterion):
    model.train()
    total_loss = 0
    for data_inter, data_intra1, data_intra2 in zip(data_loader_inter, data_loader_intra1, data_loader_intra2):
        data_inter = data_inter.to(device)
        data_intra1 = data_intra1.to(device)
        data_intra2 = data_intra2.to(device)

        optimizer.zero_grad()

        out = model(data_inter, data_intra1, data_intra2)
        out = torch.squeeze(out)
        label = data_inter.y.to(device)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_inter.num_graphs
    return total_loss / len(data_loader_inter.dataset)


def evaluate(model, data_loader_inter, data_loader_intra1, data_loader_intra2, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data_inter, data_intra1, data_intra2 in zip(data_loader_inter, data_loader_intra1, data_loader_intra2):
            data_inter = data_inter.to(device)
            data_intra1 = data_intra1.to(device)
            data_intra2 = data_intra2.to(device)

            out = model(data_inter, data_intra1, data_intra2)
            out = torch.squeeze(out)
            label = data_inter.y.to(device)
            loss = criterion(out, label)
            total_loss += loss.item() * data_inter.num_graphs
    return total_loss / len(data_loader_inter.dataset)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_channels = datalist_inter[0].num_node_features
hidden_channels = 256
out_channels = 64
linear_out1 = 32   
linear_out2 = 1
edge_dim = datalist_inter[0].num_edge_features
num_layers = 3 
num_timesteps = 2
dropout = 0.5


model = GraphNetwork(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2).to(device)

# Load the trained parameters
model.load_state_dict(torch.load('model/model_trained.pkl'))
model.eval()  # Set the model to evaluation mode

all_predictions = []
all_true_values = []

with torch.no_grad():

# criterion = torch.nn.MSELoss()
    criterion_test = torch.nn.L1Loss()
    test_loss = evaluate(model, test_loader_inter, test_loader_intra1, test_loader_intra2, criterion=criterion_test)
    test_loss_ref = test_loss * 1.364
    print(f'L1 Test Loss: {test_loss:.3f}')
    print(f'Ref Test Loss: {test_loss_ref:.3f}')

    # Disable gradient computation during testing
    for batch_inter, batch_intra1, batch_intra2 in zip(test_loader_inter, test_loader_intra1, test_loader_intra2):

        batch_inter = batch_inter.to(device)
        batch_intra1 = batch_intra1.to(device)
        batch_intra2 = batch_intra2.to(device)

        y_true = batch_inter.y
        y_true = y_true.to(device)
        
        # Get model predictions for the current batch
        y_pred = model(batch_inter, batch_intra1, batch_intra2)
        y_pred = torch.squeeze(y_pred)
        
        # Store predictions and true values
        all_predictions.append(y_pred.cpu().numpy())
        all_true_values.append(y_true.cpu().numpy())


for i in range(len(all_predictions)):
    # Check if the current item is a scalar by examining its dimensionality
    if all_predictions[i].ndim == 0:
        # Convert scalar to a 1D array and update the item in the list
        all_predictions[i] = np.array([all_predictions[i]])

for i in range(len(all_true_values)):
    # Check if the current item is a scalar by examining its dimensionality
    if all_true_values[i].ndim == 0:
        # Convert scalar to a 1D array and update the item in the list
        all_true_values[i] = np.array([all_true_values[i]])

# Concatenate all predictions and true values
all_predictions = np.concatenate(all_predictions, axis=0)
all_true_values = np.concatenate(all_true_values, axis=0)

pearson_coefficient = np.corrcoef(all_true_values, all_predictions)[0, 1]
print(f'Pearson Correlation Coefficient: {pearson_coefficient:.3f}')




