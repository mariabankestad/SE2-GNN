import torch
from torch import nn
from torch_geometric.utils import softmax
from .util import MLP

class TransformerMessage(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_channels = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = node_dim*3
        self.linear_in = torch.nn.Linear(node_dim*2 + edge_dim, hidden_channels)
        self.act_alpha = nn.LeakyReLU(inplace=True)
        self.linear_alpha = nn.Linear(hidden_channels,1)
        self.linear_mess = nn.Linear(hidden_channels, node_dim)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.act_mess = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_in.reset_parameters()
        self.layer_norm.reset_parameters()
        self.linear_alpha.reset_parameters()
        self.linear_mess.reset_parameters()

    def forward(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        _,col = edge_index
        mess = torch.cat([x_j, x_i, edge_attr], dim = 1)
        mess = self.linear_in(mess)

        alpha = self.act_alpha(self.layer_norm(mess))
        
        alpha = self.linear_alpha(alpha)
        alpha = softmax(alpha, col, num_nodes=num_nodes).reshape(-1,1)
        mess = self.linear_mess(self.act_mess(mess))
        mess = mess*alpha

        return mess

class MLPMessage(nn.Module):
    def __init__(self, node_dim, edge_dim, dim_out = None):
        super().__init__()
        
        if dim_out is None:
            dim_out = node_dim
        hidden_channels = node_dim*3
        self.mlp = MLP(node_dim*2 + edge_dim, [hidden_channels] ,dim_out)

    def reset_parameters(self):
        self.mlp.reset_parameters()


    def forward(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        _,col = edge_index
        mess = torch.cat([x_j, x_i, edge_attr], dim = 1)
        return self.mlp(mess)