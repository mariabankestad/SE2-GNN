from torch import nn
import torch
from .util import MLP, get_rot, get_rot_mulitK, rotate
from .so2norm import SO2LayerNorm
from .so2layers import SO2MLP      
from .layers import EqLayer, InvLayer, EqLayerLoc, EqLayerNodeAttr, InvMPLayer, EqLayerSimple, EqLayerESCNN
from .ChebConv import ChebConvRot
from torch_geometric.nn.pool import knn_graph

from .message_layers import MLPMessage


class FeedForward(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars, N = 128, pointwise = False):
        super().__init__() 
        self.so2_mlp = SO2MLP(L_max, num_rep, num_rep*3, N = N, pointwise = pointwise)
        self.scalar_mlp = MLP(n_scalars,[n_scalars*2], n_scalars) 

    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        x_rot = self.so2_mlp(x_rot, rot_theta)
        x_scalars = self.scalar_mlp(x_scalars)
        return {"scalar": x_scalars, "rot": x_rot}

class FeedForwardRot(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars):
        super().__init__() 
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.num_rep = num_rep

        dim = n_scalars + num_rep*L_max*2
        self.mlp = MLP(dim ,[dim*3], dim, activation='leaky_relu') 

    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x_rot.shape[:-1] + (self.L_max,2)
        x_rot = torch.einsum("njkm, nkml -> njkl",x_rot.view(size), rot_theta_inv).flatten(start_dim=1)
        x = torch.cat([x_scalars,x_rot], dim = 1)
        x = self.mlp(x)
        x_scalars = x[:,:self.n_scalars]
        x_rot = x[:,self.n_scalars:].reshape(x_rot.size(0), self.num_rep, self.L_max*2)
        x_rot = torch.einsum("njkm, nkml -> njkl", x_rot.view(size), rot_theta).flatten(start_dim=2)
        return {"scalar": x_scalars, "rot": x_rot}
    

    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1, rot_theta1)
        out2 = self.forward(input2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))
           
class FeedForwardInv(nn.Module):
    def __init__(self, n_scalars):
        super().__init__() 
        #self.so2_mlp = SO2MLP(L_max, num_rep, num_rep*3)
        self.n_scalars = n_scalars

        self.mlp = MLP(n_scalars ,[n_scalars*3], n_scalars, activation='leaky_relu') 

    def forward(self, x):
        x = self.mlp(x)
        return x
    
class EqBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep, rot_feed_forward = False, N = 128, pointwise = False):
        super().__init__()  

        self.conv_layer = EqLayer( dist_emb_dim,n_scalars,num_rep, L_max)
        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars, N = N, pointwise = pointwise)
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,dist_embedding, rot) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
        
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))


class EqBlockSimple(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep, rot_feed_forward = False, N = 128, pointwise = False):
        super().__init__()  

        self.conv_layer = EqLayerSimple( dist_emb_dim,n_scalars,num_rep, L_max)
        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars, N = N, pointwise = pointwise)
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,dist_embedding, rot) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
        
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

class EqMLPBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep, rot_feed_forward = False):
        super().__init__()  

        self.conv_layer = EqLayer( dist_emb_dim,n_scalars,num_rep, L_max,message_function=MLPMessage)
        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars )
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,dist_embedding, rot) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

class EqMLPNodeAttrBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep,n_scalars_node_attr,num_rep_node_attr, rot_feed_forward = False):
        super().__init__()  

        self.conv_layer = EqLayerNodeAttr( dist_emb_dim,n_scalars,num_rep, L_max,n_scalars_node_attr,num_rep_node_attr,message_function=MLPMessage)
        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars )
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, node_attr, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, node_attr, edge_index,dist_embedding, rot) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))


class ChebBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep, rot_feed_forward = False):
        super().__init__()  
    
        self.conv_layer = ChebConvRot( dist_emb_dim,5, L_max, n_scalars,num_rep )

        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars )
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index, rot_theta, dist_embedding, batch):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,rot_theta, dist_embedding, batch) 

    
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))


class EqBlockLoc(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, L_max, num_rep, rot_feed_forward = False):
        super().__init__()  

        self.conv_layer = EqLayerLoc( dist_emb_dim,n_scalars,num_rep, L_max)
        self.rot_feed = rot_feed_forward
        if rot_feed_forward:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        else:
            self.feed_forwad = FeedForward(L_max, num_rep, n_scalars )
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, node_attr, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, node_attr, edge_index,dist_embedding, rot) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))


class InvBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars):
        super().__init__()  

        self.conv_layer = InvLayer( dist_emb_dim,n_scalars)
        self.feed_forwad = FeedForwardInv(n_scalars )
        self.layer_norm1 = torch.nn.LayerNorm(n_scalars)
        self.layer_norm2 = torch.nn.LayerNorm(n_scalars)
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,dist_embedding) 
        x = x + x_

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_) 
        x = x + x_

        return x
    

class InvMPBlock(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars):
        super().__init__()  

        self.conv_layer = InvMPLayer( dist_emb_dim,n_scalars)
        self.feed_forwad = FeedForwardInv(n_scalars )
        self.layer_norm1 = torch.nn.LayerNorm(n_scalars)
        self.layer_norm2 = torch.nn.LayerNorm(n_scalars)
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding):
        # x must be a geoemtric tensor

        x_ = self.layer_norm1(x)
        x_ = self.conv_layer( x_, edge_index,dist_embedding) 
        x = x + x_

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_) 
        x = x + x_

        return x
    

from .so2layers import SO2MLP_escnn

class FeedForwardESCNN(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars, device = "cuda"):
        super().__init__() 
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.num_rep = num_rep
        self.mlp_scaras = MLP(n_scalars ,[n_scalars*3], n_scalars, activation='leaky_relu') 

        self.mlp_rot = SO2MLP_escnn(num_rep, num_rep*3, num_rep, device = device)

    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        x_scalars = self.mlp_scaras(x_scalars)
        x_rot = self.mlp_rot(x_rot)
        return {"scalar": x_scalars, "rot": x_rot}
    

    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1, rot_theta1)
        out2 = self.forward(input2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))



class EqBlockSimple2(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, escnn_feed_forward = False,escnn_conv = False, device = "cuda"):
        super().__init__()  

        L_max = 1
        if escnn_conv:
            self.conv_layer = EqLayerESCNN( dist_emb_dim,n_scalars,num_rep, L_max)
        else:
            self.conv_layer = EqLayerSimple( dist_emb_dim,n_scalars,num_rep, L_max)
        if escnn_feed_forward:
            self.feed_forwad = FeedForwardESCNN(L_max, num_rep, n_scalars,device = device )
        else:
            self.feed_forwad = FeedForwardRot(L_max, num_rep, n_scalars )
        self.layer_norm1 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.layer_norm2 = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.dist_emb_dim = dist_emb_dim


    def forward(self, x, edge_index,dist_embedding, rot, rot_theta = None):
        # x must be a geoemtric tensor
        x_ = self.layer_norm1(x)

        x_ = self.conv_layer( x_, edge_index,dist_embedding, rot) 
        
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        x_ = self.layer_norm2(x)
        x_ = self.feed_forwad(x_, rot_theta) 
        x['scalar'] = x['scalar'] + x_['scalar']
        x['rot'] = x['rot'] + x_['rot']

        return x
        
    
    def test_equivariance(self, theta = 0.33):
        N = 12
        x_rot = torch.randn(N,self.num_rep, self.L_max*2)
        x_scal = torch.randn(N,self.n_scalars)

        pos = torch.randn(N,2)
        pos = pos-pos.mean(dim = 0)

        edge_index = knn_graph(pos, k = 6)

        row,col = edge_index
        dist_emb = torch.randn(len(row),self.dist_emb_dim)
        
        edge_vec1 = pos[row]- pos[col]
        rot1 = get_rot(edge_vec1,self.L_max)

        pos2 = rotate(theta, pos)

        rot_theta1 = get_rot(pos,self.L_max)
        rot_theta2 = get_rot(pos2,self.L_max)

        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    

        out1 = self.forward(input1,edge_index,dist_emb,rot1, rot_theta1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2, rot_theta2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))