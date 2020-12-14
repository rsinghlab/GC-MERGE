"""
run_model.py

Purpose: Predict gene expression given graph structure and node features

Usage: python ./run_model.py [-c <str>] [-e <int>] [-lr <float>] [-cn <int]
            [-gs <int>] [-ln <int>] [-ls <int>]

Arguments:
    '-c', '--cell_line', default='E116', type=str)
    '-e', '--max_epoch', default=1000,type=int) 
    '-lr', '--learning_rate', default=1e-4, type=float)
    '-cn', '--num_graph_conv_layers', default=2, type=int)
    '-gs', '--graph_conv_embed_size', default=128, type=int)
    '-ln', '--num_lin_layers', default=2, type=int)
    '-ls', '--lin_hidden_layer_size', default=128, type=int)

Processed inputs: 
    In ./data/cell_line subdirectory:
        ./hic_sparse.npz: Concatenated Hi-C matrix in sparse CSR format
        ./np_nodes_lab_genes.npy: Numpy array stored in binary format
            2-column array that stores IDs of nodes corresponding to genes
            and the node label (expression level)
        ./np_hmods_norm.npy: Numpy array stored in binary format
            (F+1)-column array where the 0th column contains node IDs
            and columns 1..F contain feature values, where F = total number of features
        ./df_genes.pkl: Pandas dataframe stored in .pkl format
            5-column dataframe, where columns = [ENSEMBL ID, 
            gene name abbreviation, node ID, expression level, connected status]
    *Note: Users can prepare these files or use process_inputs.py script provided

Outputs:  
    In ./data/cell_line/saved_runs subdirectory:
        model_[date_and_time].pt: Model state dictionary stored in .pt (PyTorch) format
        model_predictions_[date_and_time].csv: Predictions for each gene
            Columns: [Dataset,Node ID, ENSEMBL ID, gene name abbreviation,
                true label, predicted label, classification [TP/TN/FP/FN]]
        model_[date_and_time]_info.txt: Text file containing summary of model
            statistics (test AUC, F1 scores) as well as hyperparameter settings

"""

import os
import argparse
import time
from datetime import datetime, date
import random

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd

import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn

from sage_conv_cat import SAGEConvCat


class GCN(nn.Module):

    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_embed_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_embed_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted (2)

        Returns
        -------
        None.

        '''
        
        super(GCN, self).__init__()
        
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout_value = 0
        
        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes[0])
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes[0])
            self.conv2 = SAGEConvCat(graph_conv_embed_sizes[0], graph_conv_embed_sizes[1])
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(num_feat, graph_conv_embed_sizes[0])
            self.conv2 = SAGEConvCat(graph_conv_embed_sizes[0], graph_conv_embed_sizes[1])
            self.conv3 = SAGEConvCat(graph_conv_embed_sizes[1], graph_conv_embed_sizes[2])
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(graph_conv_embed_sizes[-1], num_classes)
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(graph_conv_embed_sizes[-1], lin_hidden_sizes[0])
            self.lin2 = nn.Linear(lin_hidden_sizes[0], num_classes)
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(graph_conv_embed_sizes[-1], lin_hidden_sizes[0])
            self.lin2 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin3 = nn.Linear(lin_hidden_sizes[1], num_classes)
    
        self.loss_calc = nn.CrossEntropyLoss()
        self.torch_softmax = nn.Softmax(dim=1)

        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function.
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Un-normalized class scores

        '''
        if self.num_graph_conv_layers == 1:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 2:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
        elif self.num_graph_conv_layers == 3:
            h = self.conv1(x, edge_index)
            h = torch.relu(h)
            h = self.conv2(h, edge_index)
            h = torch.relu(h)
            h = self.conv3(h, edge_index)
            h = torch.relu(h)
                    
        dropout_value = 0.5
        
        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = F.dropout(scores, p = dropout_value, training=train_status)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = F.dropout(scores, p = dropout_value, training=train_status)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        return scores
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Un-normalized class scores from forward function
        labels [tensor]: Class labels for nodes

        Returns
        -------
        xent_loss [tensor]: Cross-entropy loss

        '''

        xent_loss = self.loss_calc(scores, labels)

        return xent_loss
    
    def calc_softmax_pred(self, scores):
        '''
        Calculates softmax scores and predicted classes

        Parameters
        ----------
        scores [tensor]: Un-normalized class scores

        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class

        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted

    
def train_model(net, graph, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer):
    '''
    Parameters
    ----------
    net [GCN]: Instantiation of model class
    graph [PyG Data class]: PyTorch Geometric Data object representing the graph
    max_epoch [int]: Maximum number of training epochs
    learning_rate [float]: Learning rate
    targetNode_mask [tensor]: Subgraph mask for training nodes
    train_idx [array]: Node IDs corresponding to training set
    valid_idx [array]: Node IDs corresponding to validation set
    optimizer [PyTorch optimizer class]: PyTorch optimization algorithm

    Returns
    -------
    train_loss_vec [array]: Training loss for each epoch
    train_AUC_vec [array]: Training AUC score for each epoch
    valid_loss_vec [array]: Validation loss for each epoch
    valid_AUC_vec [array]: Validation AUC score for each epoch

    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net.to(device)
    graph = graph.to(device)

    optimizer = optimizer
    
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    
    train_loss_list = []
    train_AUC_vec = np.zeros(np.shape(np.arange(max_epoch)))
    valid_loss_list = []
    valid_AUC_vec = np.zeros(np.shape(np.arange(max_epoch)))

    model.train()
    train_status = True
    
    print('\n')
    for e in list(range(max_epoch)):
        
        if e%100 == 0:
            print("Epoch", str(e), 'out of', str(max_epoch))
        
        model.train()
        train_status = True
        
        optimizer.zero_grad()
        
        ### Only trains on nodes with genes due to masking
        forward_scores = model(graph.x.float(), graph.edge_index, train_status)[targetNode_mask]
        
        train_scores = forward_scores[train_idx]

        train_loss  = model.loss(train_scores, torch.LongTensor(train_labels).to(device))

        train_softmax, _ = model.calc_softmax_pred(train_scores)

        train_loss.backward()
        
        optimizer.step()
            
        ### Calculate training and validation loss, AUC scores
        model.eval()
        
        valid_scores = forward_scores[valid_idx]
        valid_loss  = model.loss(valid_scores, torch.LongTensor(valid_labels).to(device))
        valid_softmax, _ = model.calc_softmax_pred(valid_scores) 

        train_loss_list.append(train_loss.item())
        train_softmax = to_cpu_npy(train_softmax)
        train_AUC = roc_auc_score(train_labels, train_softmax[:,1], average="micro")

        valid_loss_list.append(valid_loss.item())
        valid_softmax = to_cpu_npy(valid_softmax)
        valid_AUC = roc_auc_score(valid_labels, valid_softmax[:,1], average="micro")
        
        train_AUC_vec[e] = train_AUC
        valid_AUC_vec[e] = valid_AUC

    train_loss_vec = np.reshape(np.array(train_loss_list), (-1, 1))
    valid_loss_vec = np.reshape(np.array(valid_loss_list), (-1, 1))

    return train_loss_vec, train_AUC_vec, valid_loss_vec, valid_AUC_vec


def eval_model(net, graph, targetNode_mask, train_idx, valid_idx, test_idx):
    '''
    Run final model and compute evaluation statistics post-training

    Parameters
    ----------
    net [GCN]: Instantiation of model class
    graph [PyG Data class]: PyTorch Geometric Data object representing the graph
    targetNode_mask [tensor]: Mask ensuring model only trains on nodes with genes
    train_idx [array]: Node IDs corresponding to training set;
        analogous for valid_idx and test_idx

    Returns
    -------
    test_AUC [float]: Test set AUC scores;
    test_pred [array]: Test set predictions;
        analogously for train_pred (training set) and valid_pred (validation set)
    test_labels [array]: Test set labels;
        analagously for train_labels (training set) and valid_labels (validation set)
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = net.to(device)
    graph = graph.to(device)
    test_labels = to_cpu_npy(graph.y[targetNode_mask[test_idx]])
    
    model.eval()
    train_status=False

    forward_scores = model(graph.x.float(), graph.edge_index, train_status)[targetNode_mask]

    test_scores = forward_scores[test_idx]
    test_softmax, test_pred = model.calc_softmax_pred(test_scores) 
    
    test_softmax = to_cpu_npy(test_softmax)
    test_pred = to_cpu_npy(test_pred)
    test_AUC = roc_auc_score(test_labels, test_softmax[:,1], average="micro")
    test_F1 = f1_score(test_labels, test_pred, average="micro")
    
    train_scores = forward_scores[train_idx]
    train_labels = to_cpu_npy(graph.y[targetNode_mask[train_idx]])
    train_softmax, train_pred = model.calc_softmax_pred(train_scores) 
    train_pred = to_cpu_npy(train_pred)
    train_F1 = f1_score(train_labels, train_pred, average="micro")

    valid_scores = forward_scores[valid_idx]
    valid_labels = to_cpu_npy(graph.y[targetNode_mask[valid_idx]])
    valid_softmax, valid_pred = model.calc_softmax_pred(valid_scores) 
    valid_pred = to_cpu_npy(valid_pred)
    valid_F1 = f1_score(valid_labels, valid_pred, average="micro")

    return test_AUC, test_F1, test_pred, test_labels, train_F1, train_pred, train_labels, \
        valid_F1, valid_pred, valid_labels


def to_cpu_npy(x):
    '''
    Simple helper function to transfer GPU tensors to CPU numpy matrices

    Parameters
    ----------
    x [tensor]: PyTorch tensor stored on GPU

    Returns
    -------
    new_x [array]: Numpy array stored on CPU

    '''

    new_x = x.cpu().detach().numpy()
    
    return new_x


###Set options and random seed
parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cell_line', default='E116', type=str)
parser.add_argument('-e', '--max_epoch', default=1000,type=int) 
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float)
parser.add_argument('-cn', '--num_graph_conv_layers', default=2, type=int)
parser.add_argument('-gs', '--graph_conv_embed_size', default=128, type=int)
parser.add_argument('-ln', '--num_lin_layers', default=2, type=int)
parser.add_argument('-ls', '--lin_hidden_layer_size', default=128, type=int)

args = parser.parse_args()
cell_line = args.cell_line
max_epoch = args.max_epoch
learning_rate = args.learning_rate
num_graph_conv_layers = args.num_graph_conv_layers
graph_conv_embed_sz = args.graph_conv_embed_size
num_lin_layers = args.num_lin_layers
lin_hidden_size = args.lin_hidden_layer_size

random_seed = random.randint(0,10000)
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


###Initialize start time
start_time = time.time()

today = date.today()
mdy = today.strftime("%Y-%m-%d")
clock = datetime.now()
hms = clock.strftime("%H-%M-%S")
hm = clock.strftime("%Hh-%Mm")
hm_colon = clock.strftime("%H:%M")
date_and_time = mdy + '-at-' + hms


###Test for GPU availability
cuda_flag = torch.cuda.is_available()
if cuda_flag:  
  dev = "cuda" 
else:
  dev = "cpu"  
device = torch.device(dev)  


###Load input files
base_path = os.getcwd()
save_dir = os.path.join(base_path, 'data', cell_line, 'saved_runs')
hic_sparse_mat_file = os.path.join(base_path, 'data', cell_line, 'hic_sparse.npz')
np_nodes_lab_genes_file = os.path.join(base_path, 'data', cell_line, 'np_nodes_lab_genes.npy')
np_hmods_norm_all_file = os.path.join(base_path, 'data', cell_line, 'np_hmods_norm.npy') 
df_genes_file = os.path.join(base_path, 'data', cell_line, 'df_genes.pkl')
df_genes = pd.read_pickle(df_genes_file)


###Print model specifications
print(os.path.basename(__file__))
print('Model date and time:')
print(date_and_time, '\n\n')
print('Cell line:', cell_line)
print('\n')
print('Training set: 70%')
print('Validation set: 15%')
print('Testing set: 15%')
print('\n')
print('Model hyperparameters: ')
print('Number of epochs:', max_epoch)
print('Learning rate:', learning_rate)
print('Number of graph convolutional layers:', str(num_graph_conv_layers))
print('Graph convolutional embedding size:', graph_conv_embed_sz)
print('Number of linear layers:', str(num_lin_layers))
print('Linear hidden layer size:', lin_hidden_size)


###Define model inputs
num_feat = 5
num_classes = 2

mat = load_npz(hic_sparse_mat_file)
allNodes_hms = np.load(np_hmods_norm_all_file)
hms = allNodes_hms[:, 1:] #only includes features, not node ids
allNodes = allNodes_hms[:, 0].astype(int)
geneNodes_labs = np.load(np_nodes_lab_genes_file)
geneNodes = geneNodes_labs[:, -2].astype(int)
geneLabs = geneNodes_labs[:, -1].astype(int)

allLabs = 2*np.ones(np.shape(allNodes))
allLabs[geneNodes] = geneLabs

targetNode_mask = torch.tensor(geneNodes).long()
X = torch.tensor(hms).float().reshape(-1, 5)
Y = torch.tensor(allLabs).long()

extract = torch_geometric.utils.from_scipy_sparse_matrix(mat)
data = torch_geometric.data.Data(edge_index = extract[0], edge_attr = extract[1], x = X, y = Y)
G = data
num_feat = 5

graph_conv_embed_sizes = (graph_conv_embed_sz,)*num_graph_conv_layers
lin_hidden_sizes = (lin_hidden_size,)*num_lin_layers


###Randomize node order and split into 70%/15%/15% training/validation/test sets
pred_idx_shuff = torch.randperm(targetNode_mask.shape[0])

fin_train = np.floor(0.7*pred_idx_shuff.shape[0]).astype(int)
fin_valid = np.floor(0.85*pred_idx_shuff.shape[0]).astype(int)
train_idx = pred_idx_shuff[:fin_train]
valid_idx = pred_idx_shuff[fin_train:fin_valid]
test_idx = pred_idx_shuff[fin_valid:]

train_gene_ID = targetNode_mask[train_idx].numpy()
valid_gene_ID = targetNode_mask[valid_idx].numpy()
test_gene_ID = targetNode_mask[test_idx].numpy()


###Instantiate neural network model
net = GCN(num_feat, num_graph_conv_layers, graph_conv_embed_sizes, num_lin_layers, lin_hidden_sizes, num_classes)
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, net.parameters()), 
                            lr = learning_rate)

### Print model's state_dict
print("\n"+"Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())


### Train model
train_loss_vec, train_AUC_vec, valid_loss_vec, valid_AUC_vec = train_model(net, G, max_epoch, learning_rate, targetNode_mask, train_idx, valid_idx, optimizer)


### Evaluate model
test_AUC, test_F1, test_pred, test_labels, train_F1, train_pred, train_labels, \
        valid_F1, valid_pred, valid_labels = \
            eval_model(net, G, targetNode_mask, train_idx, valid_idx, test_idx)


### Save metrics and node predictions
train_metrics = [train_gene_ID, train_pred, train_labels, train_AUC_vec, train_loss_vec]

valid_metrics = [valid_gene_ID, valid_pred, valid_labels, valid_AUC_vec, valid_loss_vec]

test_metrics = [test_gene_ID, test_pred, test_labels, test_AUC, ['na']]

dataset_list = [train_metrics, valid_metrics, test_metrics]
df_full_metrics = pd.DataFrame(columns=['Dataset','Node ID','True Label','Predicted Label','Classification'])

for d in np.arange(len(dataset_list)):
    dataset_metrics = dataset_list[d]
    partial_metrics = pd.DataFrame()

    partial_metrics['Node ID'] = dataset_metrics[0]
    partial_metrics['True Label'] = dataset_metrics[2]
    partial_metrics['Predicted Label'] = dataset_metrics[1]
    partial_metrics['Classification'] = dataset_metrics[1]*1 + dataset_metrics[1]*2
    partial_metrics['Classification'].replace(to_replace=0, value='TN', inplace=True)
    partial_metrics['Classification'].replace(to_replace=1, value='FN', inplace=True)
    partial_metrics['Classification'].replace(to_replace=2, value='FP', inplace=True)
    partial_metrics['Classification'].replace(to_replace=3, value='TP', inplace=True)
    
    if d == 0:
        partial_metrics['Dataset'] = 'Training'
    elif d == 1:
        partial_metrics['Dataset'] = 'Validation'
    elif d == 2:
        partial_metrics['Dataset'] = 'Testing'

    df_full_metrics = df_full_metrics.append(partial_metrics)

df_gene_names = df_genes.iloc[:,:3]
df_gene_names = df_gene_names.rename(columns={"gene_catalog_name": "ENSEMBL_ID", "abbrev": "Abbreviation",
                              "hic_node_id" : 'Node ID'})
df_full_metrics = pd.merge(df_full_metrics, df_gene_names, how='inner', on='Node ID')
df_full_metrics = df_full_metrics[df_full_metrics.columns[[0,1,5,6,2,3,4]]]


### Print elapsed time and performance
elapsed = (time.time() - start_time)
elapsed_h = int(elapsed//3600)
elapsed_m = int((elapsed - elapsed_h*3600)//60)
elapsed_s = int(elapsed - elapsed_h*3600 - elapsed_m*60)
print('Elapsed time: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))

print('\nPerformance:')
print('test AUC:', test_AUC, '\n')
print('test F1:', test_F1, '\n\n')


### Save trained model parameters, model predictions CSV file, model performance/information
model_path = os.path.join(save_dir, 'model_' + date_and_time + '.pt')
torch.save(net.state_dict(), model_path)

df_full_metrics_filename = os.path.join(save_dir, 'model_predictions_' + date_and_time + '.csv')
df_full_metrics.to_csv(df_full_metrics_filename, index=False)

model_info_filename = os.path.join(save_dir,'model_' + date_and_time + '_info.txt')
f = open(model_info_filename, 'w')
f.write('File name: ' + os.path.basename(__file__) + '\n')
f.write('Model reference date and time: ' + date_and_time + '\n\n')
f.write('Start date: ' + mdy + '\n')
f.write('Start time: ' + hm_colon + '\n')
f.write('Total time: {0:02d}:{1:02d}:{2:02d}'.format(elapsed_h, elapsed_m, elapsed_s))
f.write('\n\n')
f.write('Cell line: ' + cell_line + '\n')
f.write('Dataset split:\n')
f.write('Training set: 70%' + '\n')
f.write('Validation set: 15%' + '\n')
f.write('Testing set: 15%' + '\n\n')
f.write('Performance:\n')
f.write('Testing AUC: ' + str(test_AUC) + '\n')
f.write('Testing F1: ' + str(test_F1) + '\n')
f.write('Hyperparameters:\n')
f.write('Number of epochs: ' + str(max_epoch) + '\n')
f.write('Learning rate :' + str(learning_rate) + '\n')
f.write('Number of graph convolutional layers: ' + str(num_graph_conv_layers) + '\n')
f.write('Graph convolutional embedding size: ' + str(graph_conv_embed_sz) + '\n')
f.write('Number of linear layers: ' + str(num_lin_layers) + '\n')
f.write('Linear hidden layer size: ' + str(lin_hidden_size) + '\n\n')
f.write('Model\'s state_dict:\n')

for param_tensor in net.state_dict():
    f.write(str(param_tensor) + "\t" + str(net.state_dict()[param_tensor].size()) + '\n')
f.close()










