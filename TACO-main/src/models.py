import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GATConv, GraphConv, GINConv
import math

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATConv(input_dim, hidden_dim, num_heads)
        self.layer2 = GATConv(hidden_dim, output_dim, 1)
        self.return_hidden = False
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g, features):
        x1 = F.relu(self.layer1(g, features.float()))
        x1 = torch.mean(x1, 1)
        x2 = self.layer2(g, x1.float()).squeeze()
        if self.return_hidden == False:
            return x2
        return x1

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.layer1 = GraphConv(input_dim, hidden_dim)
        self.layer2 = GraphConv(hidden_dim, output_dim)
        self.return_hidden = False
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g, features):

        x1 = F.relu(self.layer1(g, features))
        x2 = self.layer2(g, x1)
        if self.return_hidden == False:
            return x2
        return x1

class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.layer1 = GINConv(torch.nn.Linear(input_dim, hidden_dim), 'mean' )
        self.layer2 = GINConv(torch.nn.Linear(hidden_dim, output_dim), 'mean')
        self.return_hidden = False
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data, gain=nn.init.calculate_gain('relu'))
            else:
                stdv = 1. / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, g, features):
        g = dgl.graph(g.edges())
        x1 = F.relu(self.layer1(g, features.float()))
        x2 = self.layer2(g, x1)
        if self.return_hidden == False:
            return x2
        return x1
