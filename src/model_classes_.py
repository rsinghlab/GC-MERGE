import os
import argparse
import time
from datetime import datetime, date
import random

import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from scipy.stats import pearsonr
import pandas as pd

import torch
import torch_geometric
import torch.nn.functional as F
import torch.nn as nn

from sage_conv_cat_ import SAGEConvCat


class GCN_regression(nn.Module):

    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_layer_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines regression model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Size of predicted output tensor for batch size of N, 
            i.e. N x num_classes(=1)

        Returns
        -------
        None.

        '''
        
        super(GCN_regression, self).__init__()
        
        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout = 0.5
    
        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
            self.conv3 = SAGEConvCat(graph_conv_layer_sizes[2], graph_conv_layer_sizes[3])
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
        
        self.loss_calc = nn.MSELoss()

        
    def forward(self, x, edge_index, train_status=False):
        '''
        Forward function
        
        Parameters
        ----------
        x [tensor]: Node features
        edge_index [tensor]: Subgraph mask
        train_status [bool]: optional, set to True for dropout

        Returns
        -------
        scores [tensor]: Predicted expression levels

        '''
        ### Graph convolution module
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

        h = F.dropout(h, p = self.dropout, training=train_status)

        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        if len(scores.size()) > 1:
            scores = scores.squeeze()
            
        return scores
    
    
    def loss(self, scores, targets):
        '''
        Calculates mean squared error loss
        
        Parameters
        ----------
        scores [tensor]: Predicted scores from forward function
        labels [tensor]: Target scores 

        Returns
        -------
        mse [tensor]: Mean squared error loss

        '''
        
        mse = self.loss_calc(scores, targets)

        return mse
    

class GCN_classification(nn.Module):

    def __init__(self, num_feat, num_graph_conv_layers, graph_conv_layer_sizes, num_lin_layers, lin_hidden_sizes, num_classes):
        '''
        Defines classification model class

        Parameters
        ----------
        num_feat [int]: Feature dimension (int)
        num_graph_conv_layers [int]: Number of graph convolutional layers (1, 2, or 3)
        graph_conv_layer_sizes [int]: Embedding size of graph convolutional layers 
        num_lin_layers [int]: Number of linear layers (1, 2, or 3)
        lin_hidden_sizes [int]: Embedding size of hidden linear layers
        num_classes [int]: Number of classes to be predicted(=2)

        Returns
        -------
        None.

        '''
        
        super(GCN_classification, self).__init__()

        self.num_graph_conv_layers = num_graph_conv_layers
        self.num_lin_layers = num_lin_layers
        self.dropout_value = 0.5

        if self.num_graph_conv_layers == 1:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
        elif self.num_graph_conv_layers == 2:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
        elif self.num_graph_conv_layers == 3:
            self.conv1 = SAGEConvCat(graph_conv_layer_sizes[0], graph_conv_layer_sizes[1])
            self.conv2 = SAGEConvCat(graph_conv_layer_sizes[1], graph_conv_layer_sizes[2])
            self.conv3 = SAGEConvCat(graph_conv_layer_sizes[2], graph_conv_layer_sizes[3])
        
        if self.num_lin_layers == 1:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
        elif self.num_lin_layers == 2:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
        elif self.num_lin_layers == 3:
            self.lin1 = nn.Linear(lin_hidden_sizes[0], lin_hidden_sizes[1])
            self.lin2 = nn.Linear(lin_hidden_sizes[1], lin_hidden_sizes[2])
            self.lin3 = nn.Linear(lin_hidden_sizes[2], lin_hidden_sizes[3])
            
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
        scores [tensor]: Pre-normalized class scores

        '''

        ### Graph convolution module
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
            
        h = F.dropout(h, p = self.dropout_value, training=train_status)

        ### Linear module
        if self.num_lin_layers == 1:
            scores = self.lin1(h)
        elif self.num_lin_layers == 2:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
        elif self.num_lin_layers == 3:
            scores = self.lin1(h)
            scores = torch.relu(scores)
            scores = self.lin2(scores)
            scores = torch.relu(scores)
            scores = self.lin3(scores)
        
        return scores
    
    
    def loss(self, scores, labels):
        '''
        Calculates cross-entropy loss
        
        Parameters
        ----------
        scores [tensor]: Pre-normalized class scores from forward function
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
        scores [tensor]: Pre-normalized class scores

        Returns
        -------
        softmax [tensor]: Probability for each class
        predicted [tensor]: Predicted class

        '''
        
        softmax = self.torch_softmax(scores)
        
        predicted = torch.argmax(softmax, 1)
        
        return softmax, predicted
