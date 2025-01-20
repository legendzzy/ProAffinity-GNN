import os
import torch
import pickle
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
from torch_geometric.data import DataLoader
from torch_geometric.nn.models import AttentiveFP
from torch.optim import Adam
from torch.nn import MSELoss
from torch.nn import L1Loss
import torch.nn.functional as F
import matplotlib.pyplot as plt

filter_range = False
datalist_inter = []
datalist_intra1 = []
datalist_intra2 = []

path = 'data/graph/inter_graph/'
filenames = os.listdir(path)
filenames.sort()
print(filenames)

for filename in filenames:
    pdb = filename
    f_path1 = 'data/graph/inter_graph/' + pdb
    f_path2 = 'data/graph/individual_graph/' + pdb + '_1'
    f_path3 = 'data/graph/individual_graph/' + pdb + '_2'

    print(f_path1)

    with open(f_path1, 'rb') as f_graph1:

        graph = pickle.load(f_graph1)
        graph.edge_attr = graph.edge_attr.float()

        if filter_range:
            y = graph.y
            if y > 3.0 and y < 13.0:
                datalist_inter.append(graph)

                with open(f_path2, 'rb') as f_graph2:
                    graph = pickle.load(f_graph2)
                    graph.edge_attr = graph.edge_attr.float()
                    datalist_intra1.append(graph)
                with open(f_path3, 'rb') as f_graph3:
                    graph = pickle.load(f_graph3)
                    graph.edge_attr = graph.edge_attr.float()
                    datalist_intra2.append(graph)
        else:
            datalist_inter.append(graph)
            with open(f_path2, 'rb') as f_graph2:
                graph = pickle.load(f_graph2)
                graph.edge_attr = graph.edge_attr.float()
                datalist_intra1.append(graph)
            with open(f_path3, 'rb') as f_graph3:
                graph = pickle.load(f_graph3)
                graph.edge_attr = graph.edge_attr.float()
                datalist_intra2.append(graph)

# set ramdom seed to each shuffle
random.seed(20)

combined = list(zip(datalist_inter, datalist_intra1, datalist_intra2))
random.shuffle(combined)
datalist_inter, datalist_intra1, datalist_intra2 = zip(*combined)
datalist_inter = list(datalist_inter)
datalist_intra1 = list(datalist_intra1)
datalist_intra2 = list(datalist_intra2)

data_len1 = len(datalist_inter)
data_len2 = len(datalist_intra1)
data_len3 = len(datalist_intra2)

train_set_inter = datalist_inter[:int(0.8 * data_len1)]
val_set_inter = datalist_inter[int(0.8 * data_len1):]
train_loader_inter = DataLoader(train_set_inter, batch_size=16)
val_loader_inter = DataLoader(val_set_inter, batch_size=16)

train_set_intra1 = datalist_intra1[:int(0.8 * data_len2)]
val_set_intra1 = datalist_intra1[int(0.8 * data_len2):]
train_loader_intra1 = DataLoader(train_set_intra1, batch_size=16)
val_loader_intra1 = DataLoader(val_set_intra1, batch_size=16)

train_set_intra2 = datalist_intra2[:int(0.8 * data_len3)]
val_set_intra2 = datalist_intra2[int(0.8 * data_len3):]
train_loader_intra2 = DataLoader(train_set_intra2, batch_size=16)
val_loader_intra2 = DataLoader(val_set_intra2, batch_size=16)


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
ep = 25
weight_decay = 0.001
lr = 0.0005
model_save = 'model/model.pkl'
fig_save = 'model/result.png'

# Instantiate the model
model = GraphNetwork(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()  # Use mean squared error loss for regression

# Initialize lists to record training and validation losses
train_losses = []
val_losses = []

best_loss = float('inf')
patience = 10  # wait for 10 epochs
patience_counter = 0

for epoch in range(ep):  
    train_loss = train(model, train_loader_inter, train_loader_intra1, train_loader_intra2, optimizer=optimizer, criterion=criterion)
    train_losses.append(train_loss)
    
    val_loss = evaluate(model, val_loader_inter, val_loader_intra1, val_loader_intra2, criterion=criterion)
    val_losses.append(val_loss)

    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}')
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        # Save model state if desired
        torch.save(model.state_dict(), model_save)
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"Stopping early at epoch {epoch + 1}")
        break

# Plot the training and validation losses
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.savefig(fig_save)
plt.show()

import numpy as np

# Assuming model is your GNN model and dataloader is your test dataloader
model.load_state_dict(torch.load(model_save))
model.eval()  # Set the model to evaluation mode

all_predictions = []
all_true_values = []

with torch.no_grad():

# criterion = torch.nn.MSELoss()
    criterion_test = torch.nn.L1Loss()
    test_loss = evaluate(model, val_loader_inter, val_loader_intra1, val_loader_intra2, criterion=criterion_test)
    test_loss_ref = test_loss * 1.364
    print(f'L1 Test Loss: {test_loss:.3f}')
    print(f'Ref Test Loss: {test_loss_ref:.3f}')

    # Disable gradient computation during testing
    batch_number = 0
    for batch_inter, batch_intra1, batch_intra2 in zip(val_loader_inter, val_loader_intra1, val_loader_intra2):

        batch_number += 1
        print(batch_number)
        # Assuming batch contains input data 'x' and true values 'y_true'
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

# print(all_predictions)
# Concatenate all predictions and true values
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

        
all_predictions = np.concatenate(all_predictions, axis=0)
all_true_values = np.concatenate(all_true_values, axis=0)

pearson_coefficient = np.corrcoef(all_true_values, all_predictions)[0, 1]
print(f'Pearson Correlation Coefficient: {pearson_coefficient:.3f}')

from itertools import chain
import copy
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

in_channels = datalist_inter[0].num_node_features
hidden_channels = 256
out_channels = 64

linear_out1 = 32   
linear_out2 = 1

edge_dim = datalist_inter[0].num_edge_features
num_layers = 3 
num_timesteps = 2
dropout = 0.5
ep = 25
weight_decay = 0.001
lr = 0.0005
# fig_save = 'model/'

data_len = len(datalist_inter)
set1_inter = datalist_inter[: int(0.2 * data_len)]
set2_inter = datalist_inter[int(0.2 * data_len) : int(0.4 * data_len)]
set3_inter = datalist_inter[int(0.4 * data_len) : int(0.6 * data_len)]
set4_inter = datalist_inter[int(0.6 * data_len) : int(0.8 * data_len)]
set5_inter = datalist_inter[int(0.8 * data_len) :]
set_list_inter = [set1_inter, set2_inter, set3_inter, set4_inter, set5_inter]

set1_intra1 = datalist_intra1[: int(0.2 * data_len)]
set2_intra1 = datalist_intra1[int(0.2 * data_len) : int(0.4 * data_len)]
set3_intra1 = datalist_intra1[int(0.4 * data_len) : int(0.6 * data_len)]
set4_intra1 = datalist_intra1[int(0.6 * data_len) : int(0.8 * data_len)]
set5_intra1 = datalist_intra1[int(0.8 * data_len) :]
set_list_intra1 = [set1_intra1, set2_intra1, set3_intra1, set4_intra1, set5_intra1]

set1_intra2 = datalist_intra2[: int(0.2 * data_len)]
set2_intra2 = datalist_intra2[int(0.2 * data_len) : int(0.4 * data_len)]
set3_intra2 = datalist_intra2[int(0.4 * data_len) : int(0.6 * data_len)]
set4_intra2 = datalist_intra2[int(0.6 * data_len) : int(0.8 * data_len)]
set5_intra2 = datalist_intra2[int(0.8 * data_len) :]
set_list_intra2 = [set1_intra2, set2_intra2, set3_intra2, set4_intra2, set5_intra2]


cvtrainloss = []
cvvalloss = []
pcc_list = []
l1_losses = []

for i in range(5):

    val_set_inter = set_list_inter[i]
    train_set_lists_inter = [set_list_inter[j] for j in range(5) if j != i]
    train_set_inter = list(chain(*train_set_lists_inter))
    print(len(train_set_inter))
    print(len(val_set_inter))
    train_loader_inter = DataLoader(train_set_inter, batch_size=16)
    val_loader_inter = DataLoader(val_set_inter, batch_size=16)

    val_set_intra1 = set_list_intra1[i]
    train_set_lists_intra1 = [set_list_intra1[j] for j in range(5) if j != i]
    train_set_intra1 = list(chain(*train_set_lists_intra1))
    print(len(train_set_intra1))
    print(len(val_set_intra1))
    train_loader_intra1 = DataLoader(train_set_intra1, batch_size=16)
    val_loader_intra1 = DataLoader(val_set_intra1, batch_size=16)

    val_set_intra2 = set_list_intra2[i]
    train_set_lists_intra2 = [set_list_intra2[j] for j in range(5) if j != i]
    train_set_intra2 = list(chain(*train_set_lists_intra2))
    print(len(train_set_intra2))
    print(len(val_set_intra2))
    train_loader_intra2 = DataLoader(train_set_intra2, batch_size=16)
    val_loader_intra2 = DataLoader(val_set_intra2, batch_size=16)
    
    # Instantiate the model
    model = GraphNetwork(in_channels, hidden_channels, out_channels, edge_dim, num_layers, num_timesteps, dropout, linear_out1, linear_out2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.MSELoss()  # Use mean squared error loss for regression
    #criterion = torch.nn.L1Loss()

    model_save = 'model/model_cv' + '_' + str(i+1) + '.pkl'
    # Initialize lists to record training and validation losses
    train_losses = []
    val_losses = []

    best_loss = float('inf')
    patience = 10  # for example, wait for 10 epochs
    patience_counter = 0

    for epoch in range(ep):  
        train_loss = train(model, train_loader_inter, train_loader_intra1, train_loader_intra2, optimizer=optimizer, criterion=criterion)
        train_losses.append(train_loss)
    
        val_loss = evaluate(model, val_loader_inter, val_loader_intra1, val_loader_intra2, criterion=criterion)
        val_losses.append(val_loss)

        print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Validation Loss: {val_loss:.3f}')
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save model state if desired
            torch.save(model.state_dict(), model_save)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stopping early at epoch {epoch + 1}")
            break

    # Plot the training and validation losses
    # plt.figure(figsize=(10, 5))
    # plt.plot(train_losses, label='Train Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Losses')
    # plt.legend()
    # plt.savefig(fig_save + '_cv' + str(i) + '.png')

    cvtrainloss.append(train_losses)
    cvvalloss.append(val_losses)

    # Assuming model is your GNN model and dataloader is your test dataloader
    model.load_state_dict(torch.load(model_save))
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        
        criterion_test = torch.nn.L1Loss()
        test_loss = evaluate(model, val_loader_inter, val_loader_intra1, val_loader_intra2, criterion=criterion_test)
        test_loss_ref = test_loss * 1.364
        print(f'L1 Test Loss: {test_loss:.3f}')
        print(f'Ref Test Loss: {test_loss_ref:.3f}')
        l1_losses.append(test_loss)

        for batch_inter, batch_intra1, batch_intra2 in zip(val_loader_inter, val_loader_intra1, val_loader_intra2):
            # Assuming batch contains input data 'x' and true values 'y_true'
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

    # Concatenate all predictions and true values
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_values = np.concatenate(all_true_values, axis=0)

    pearson_coefficient = np.corrcoef(all_true_values, all_predictions)[0, 1]
    pcc_list.append(pearson_coefficient)

    print(f'Pearson Correlation Coefficient: {pearson_coefficient:.3f}')


cvtrain = 0
cvval = 0
for i in range(len(cvtrainloss)):
    cvtrain += cvtrainloss[i][-1]
    cvval += cvvalloss[i][-1]

cvtrain = cvtrain / len(cvtrainloss)
cvval = cvval / len(cvvalloss)
print(l1_losses)
l1_loss = np.mean(l1_losses)


print(cvtrain, cvval)

test_loss_ref = l1_loss * 1.364
print(f'cvtrainloss: {cvtrain:.3f}, cvvalloss: {cvval:.3f}')
print(f'L1 Test Loss: {l1_loss:.3f}')
print(f'Ref Test Loss: {test_loss_ref:.3f}')

pcc = np.mean(pcc_list)
print(f'R: {pcc:.3f}')


