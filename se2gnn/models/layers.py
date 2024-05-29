from torch import nn
import torch
from .message_layers import TransformerMessage, MLPMessage
from .so2layers import SO2MLP      
from .util import MLP, rotate,get_rot_mulitK, get_rot
from torch_geometric.utils import softmax, scatter

from torch_geometric.nn.pool import knn_graph
r"""
class FeedForward(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars):
        super().__init__() 
        self.so2_mlp = SO2MLP(L_max, num_rep, num_rep*3)
        self.scalar_mlp = MLP(n_scalars,[n_scalars*2], n_scalars) 

    def forward(self, x):
        x_scalars = x['scalar']
        x_rot = x['rot']
        x_rot = self.so2_mlp(x_rot)
        x_scalars = self.scalar_mlp(x_scalars)
        return {"scalar": x_scalars, "rot": x_rot}
"""

    

class EqLayer(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, L_max, message_function = TransformerMessage, hidden_channels = None):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim = n_scalars + num_rep*L_max*2
        self.n_scalars = n_scalars
        self.message_function = message_function(x_dim, dist_emb_dim)
        self.num_rep = num_rep
        self.L_max = L_max

        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, edge_index,distance_embedding, rot):

        row, col = edge_index
        x_scalar = x['scalar']
        x_rot = x['rot']

        row, col = edge_index
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        x_scr_rot = x_rot[row,:,:]
        x_scr_rot = torch.einsum("njkm, nkml -> njkl", x_scr_rot.view(x_scr_rot.shape[:-1] + (self.L_max,2)), rot_inv)#
        x_scr_rot = torch.cat([x_scalar[row,:],x_scr_rot.flatten(start_dim=1)], dim = 1)

        x_dst_rot = x_rot[col,:,:]
        x_dst_rot = torch.einsum("njkm, nkml -> njkl", x_dst_rot.view(x_dst_rot.shape[:-1] + (self.L_max,2)), rot_inv)
        x_dst_rot = torch.cat([x_scalar[col,:],x_dst_rot.flatten(start_dim=1)], dim = 1)


        x_out = self.message_function(x_dst_rot, x_scr_rot, distance_embedding, edge_index, x_scalar.size(0))
        mess_scalar = x_out[:,:self.n_scalars]
        mess_rot = x_out[:,self.n_scalars:].reshape(x_out.size(0), self.num_rep, self.L_max*2)

        mess_rot = torch.einsum("njkm, nkml -> njkl", mess_rot.view(mess_rot.shape[:-1] + (self.L_max,2)), rot).flatten(start_dim=2)

        mess_rot = scatter(mess_rot, col, dim = 0, dim_size=x_rot.size(0))        
        mess_scalar = scatter(mess_scalar, col, dim = 0, dim_size=x_rot.size(0))

        return {"scalar": mess_scalar, "rot": mess_rot}
    
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
        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    
        out1 = self.forward(input1,edge_index,dist_emb,rot1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

class EqLayerSimple(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, L_max):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim = n_scalars + num_rep*L_max*2 
        self.n_scalars = n_scalars
        self.message_function = MLP(x_dim + dist_emb_dim, [x_dim*3], x_dim)
        self.num_rep = num_rep
        self.L_max = L_max

        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, edge_index,distance_embedding, rot):

        row, col = edge_index
        x_scalar = x['scalar']
        x_rot = x['rot']

        row, col = edge_index
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        x_scr_rot = x_rot[row,:,:]
        x_scr_rot = torch.einsum("njkm, nkml -> njkl", x_scr_rot.view(x_scr_rot.shape[:-1] + (self.L_max,2)), rot_inv)#
        x_scr_rot = torch.cat([distance_embedding, x_scalar[row,:],x_scr_rot.flatten(start_dim=1)], dim = 1)

        #x_dst_rot = x_rot[col,:,:]
        #x_dst_rot = torch.einsum("njkm, nkml -> njkl", x_dst_rot.view(x_dst_rot.shape[:-1] + (self.L_max,2)), rot_inv)
        #x_dst_rot = torch.cat([distance_embedding, x_scalar[col,:],x_dst_rot.flatten(start_dim=1)], dim = 1)


        x_out = self.message_function(x_scr_rot)
        mess_scalar = x_out[:,:self.n_scalars]
        mess_rot = x_out[:,self.n_scalars:].reshape(x_out.size(0), self.num_rep, self.L_max*2)

        mess_rot = torch.einsum("njkm, nkml -> njkl", mess_rot.view(mess_rot.shape[:-1] + (self.L_max,2)), rot).flatten(start_dim=2)

        mess_rot = scatter(mess_rot, col, dim = 0, dim_size=x_rot.size(0))        
        mess_scalar = scatter(mess_scalar, col, dim = 0, dim_size=x_rot.size(0))

        return {"scalar": mess_scalar, "rot": mess_rot}
    
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
        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    
        out1 = self.forward(input1,edge_index,dist_emb,rot1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

from .so2layers import SO2MLP_escnn_mess
class EqLayerESCNN(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, L_max):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        self.n_scalars = n_scalars
        self.message_function = SO2MLP_escnn_mess(num_rep, num_rep*3,n_scalars + dist_emb_dim, num_rep)
        self.num_rep = num_rep
        self.L_max = L_max
        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, edge_index,distance_embedding, rot):

        row, col = edge_index
        x_scalar = x['scalar']
        x_rot = x['rot']

        row, col = edge_index

        x_scr_rot = x_rot[row,:,:]
        scalars = torch.cat([distance_embedding, x_scalar[row,:]], dim = 1)
        x_out = self.message_function(x_scr_rot, scalars)
        mess_rot = scatter(x_out, col, dim = 0, dim_size=x_rot.size(0)).reshape(-1,self.num_rep,2)        

        return {"scalar": x_scalar, "rot": mess_rot}
    
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
        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    
        out1 = self.forward(input1,edge_index,dist_emb,rot1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

class EqLayerNodeAttr(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, L_max, n_scalars_node_attr, num_rep_node_attr,message_function = TransformerMessage, hidden_channels = None):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim_in = n_scalars + n_scalars_node_attr+(num_rep+ num_rep_node_attr)*L_max*2
        x_dim_out = n_scalars + (num_rep)*L_max*2

        self.n_scalars_in = n_scalars + n_scalars_node_attr
        self.n_scalars_out = n_scalars

        self.message_function = message_function(x_dim_in, dist_emb_dim, x_dim_out)
        self.num_rep_in = num_rep + num_rep_node_attr
        self.num_rep_out = num_rep

        self.L_max = L_max

        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x,node_attr, edge_index,distance_embedding, rot):

        row, col = edge_index
        x_scalar = torch.cat([x['scalar'], node_attr['scalar']], dim = 1)
        x_rot = torch.cat([x['rot'], node_attr['rot']], dim = 1)

        row, col = edge_index
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        x_scr_rot = x_rot[row,:,:]
        x_scr_rot = torch.einsum("njkm, nkml -> njkl", x_scr_rot.view(x_scr_rot.shape[:-1] + (self.L_max,2)), rot_inv)#
        x_scr_rot = torch.cat([x_scalar[row,:],x_scr_rot.flatten(start_dim=1)], dim = 1)

        x_dst_rot = x_rot[col,:,:]
        x_dst_rot = torch.einsum("njkm, nkml -> njkl", x_dst_rot.view(x_dst_rot.shape[:-1] + (self.L_max,2)), rot_inv)
        x_dst_rot = torch.cat([x_scalar[col,:],x_dst_rot.flatten(start_dim=1)], dim = 1)


        x_out = self.message_function(x_dst_rot, x_scr_rot, distance_embedding, edge_index, x_scalar.size(0))
        mess_scalar = x_out[:,:self.n_scalars_out]
        mess_rot = x_out[:,self.n_scalars_out:].reshape(x_out.size(0), self.num_rep_out, self.L_max*2)

        mess_rot = torch.einsum("njkm, nkml -> njkl", mess_rot.view(mess_rot.shape[:-1] + (self.L_max,2)), rot).flatten(start_dim=2)

        mess_rot = scatter(mess_rot, col, dim = 0, dim_size=x_rot.size(0))        
        mess_scalar = scatter(mess_scalar, col, dim = 0, dim_size=x_rot.size(0))

        return {"scalar": mess_scalar, "rot": mess_rot}
    
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
        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    
        out1 = self.forward(input1,edge_index,dist_emb,rot1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))

class EqLayerLoc(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, num_rep, L_max, message_function = TransformerMessage, hidden_channels = None):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim = n_scalars + num_rep*L_max*2
        self.n_scalars = n_scalars
        self.message_function = message_function(x_dim, dist_emb_dim)
        self.num_rep = num_rep
        self.L_max = L_max

        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, node_attr, edge_index,distance_embedding, rot):

        row, col = edge_index
        x_scalar_d = x['scalar']
        x_scalar_s = node_attr['scalar']

        x_rot_s = node_attr['rot']
        x_rot_d = x['rot']

        row, col = edge_index
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        x_scr_rot = x_rot_s[row,:,:]

        x_scr_rot = torch.einsum("njkm, nkml -> njkl", x_scr_rot.view(x_scr_rot.shape[:-1] + (self.L_max,2)), rot_inv)#
        x_scr_rot = torch.cat([x_scalar_s[row,:],x_scr_rot.flatten(start_dim=1)], dim = 1)

        x_dst_rot = x_rot_d[col,:,:]
        x_dst_rot = torch.einsum("njkm, nkml -> njkl", x_dst_rot.view(x_dst_rot.shape[:-1] + (self.L_max,2)), rot_inv)
        x_dst_rot = torch.cat([x_scalar_d[col,:],x_dst_rot.flatten(start_dim=1)], dim = 1)


        x_out = self.message_function(x_dst_rot, x_scr_rot, distance_embedding, edge_index, x_scalar_s.size(0))
        mess_scalar = x_out[:,:self.n_scalars]
        mess_rot = x_out[:,self.n_scalars:].reshape(x_out.size(0), self.num_rep, self.L_max*2)

        mess_rot = torch.einsum("njkm, nkml -> njkl", mess_rot.view(mess_rot.shape[:-1] + (self.L_max,2)), rot).flatten(start_dim=2)

        mess_rot = scatter(mess_rot, col, dim = 0, dim_size=x_rot_s.size(0))        
        mess_scalar = scatter(mess_scalar, col, dim = 0, dim_size=x_rot_s.size(0))

        return {"scalar": mess_scalar, "rot": mess_rot}
    
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
        edge_vec2 = pos2[row]- pos2[col]
        rot2 = get_rot(edge_vec2,self.L_max)

        rot_theta = get_rot_mulitK(theta,self.L_max)

        x_rot2 = torch.einsum("njkm, nkml -> njkl", x_rot.view(x_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)

        input1 = {"scalar": x_scal, "rot": x_rot}
        input2 = {"scalar": x_scal, "rot": x_rot2}    
        out1 = self.forward(input1,edge_index,dist_emb,rot1)
        out2 = self.forward(input2,edge_index,dist_emb,rot2)
        out1_rot = out1['rot']
        out2_rot = out2['rot']

        out1_rot = torch.einsum("njkm, nkml -> njkl", out1_rot.view(out1_rot.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim= 2)
        print(torch.norm(out2_rot-out1_rot))
        print(torch.norm(out1['scalar'] - out2['scalar']))


class InvLayer(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, message_function = TransformerMessage):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim = n_scalars 
        self.n_scalars = n_scalars
        self.message_function = message_function(x_dim, dist_emb_dim)


        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, edge_index,distance_embedding):

        row, col = edge_index

        x_dst = x[col,:]
        x_scr = x[row,:]

        x_out = self.message_function(x_dst, x_scr, distance_embedding, edge_index, x.size(0))


        x_out = scatter(x_out, col, dim = 0, dim_size=x.size(0))        

        return x_out

class InvMPLayer(nn.Module):
    def __init__(self, dist_emb_dim, n_scalars, message_function = MLPMessage):
        super().__init__()

        self.dist_emb_dim = dist_emb_dim
        x_dim = n_scalars 
        self.n_scalars = n_scalars
        self.message_function = message_function(x_dim, dist_emb_dim)


        self.reset_parameters()

    def reset_parameters(self):
        self.message_function.reset_parameters()


    def forward(self, x, edge_index,distance_embedding):

        row, col = edge_index

        x_dst = x[col,:]
        x_scr = x[row,:]

        x_out = self.message_function(x_dst, x_scr, distance_embedding, edge_index, x.size(0))


        x_out = scatter(x_out, col, dim = 0, dim_size=x.size(0))        

        return x_out