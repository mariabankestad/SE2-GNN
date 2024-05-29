import torch
from torch import nn
#from torch_cluster import radius_graph
from .util import besel_linspace, get_rot, MLP, rotate, get_rot_mulitK
from .so2norm import SO2LayerNorm
from .blocks import EqBlock, InvBlock, EqBlockSimple, ChebBlock, EqMLPBlock, EqMLPNodeAttrBlock, InvMPBlock
from .embeddings import Node_embedding_pendulum, Edge_embedding_tetris,Edge_embedding_tetris_inv, Node_embedding_vel, Node_embedding_inv, Node_embedding_inv_smoke, Node_embedding_updated2, Node_embedding_smoke
from .so2layers import SO2MLP

from torch_geometric.utils import  scatter


class SO2OutU(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars, n_outputs = 1):
        super().__init__() 
        #self.so2_mlp = SO2MLP(L_max, num_rep, num_rep*3)
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.num_rep = num_rep

        dim = n_scalars + num_rep*L_max*2
        self.mlp = MLP(dim ,[dim*3], n_outputs, activation='leaky_relu') 
    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x_rot.shape[:-1] + (self.L_max,2)
        x_rot = torch.einsum("njkm, nkml -> njkl",x_rot.view(size), rot_theta_inv).flatten(start_dim=1)
        x = torch.cat([x_scalars,x_rot], dim = 1)
        x = self.mlp(x)
        return x
    
class SO2OutV(nn.Module):
    def __init__(self, L_max,num_rep, n_scalars):
        super().__init__() 
        #self.so2_mlp = SO2MLP(L_max, num_rep, num_rep*3)
        self.L_max = L_max
        self.n_scalars = n_scalars
        self.num_rep = num_rep

        dim = n_scalars + num_rep*L_max*2
        self.mlp = MLP(dim ,[dim*3], 2, activation='leaky_relu') 
    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x_rot.shape[:-1] + (self.L_max,2)
        x_rot = torch.einsum("njkm, nkml -> njkl",x_rot.view(size), rot_theta_inv).flatten(start_dim=1)
        x = torch.cat([x_scalars,x_rot], dim = 1)
        x = self.mlp(x)
        x_rot = x.unsqueeze(1).bmm(rot_theta[:,0,:]).squeeze(1)
        return x_rot
    
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

        print(torch.norm(out2-rotate(theta,out1)))

#import time
class SO2Transformer(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32, rot_feed_forward = False, updated_node_embedding = False):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlock(self.r_dim, n_scalars,L_max, num_rep, rot_feed_forward)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            
            pos = pos -pos.mean(dim = 0)
            x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force, 
                                    pos = pos, 
                                    edge_index = edge_index)
            theta = 0.4
            v1 = x1[1].detach().cpu()
            v1_rot = rotate(theta,v1)

            v2 = rotate(theta, v)
            pos2 = rotate(theta, pos)
            y_force2 = rotate(theta, y_force)
            boundary_norm2 = rotate(theta, boundary_norm)
            x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force2, 
                                    pos = pos2, 
                                    edge_index = edge_index)
            v2_rot = x2[1]
            print(torch.norm(v1_rot - v2_rot))        
#import time
class SO2MessagePassing(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32, rot_feed_forward = False, updated_node_embedding = False):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqMLPBlock(self.r_dim, n_scalars,L_max, num_rep, rot_feed_forward)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            
            pos = pos -pos.mean(dim = 0)
            x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force, 
                                    pos = pos, 
                                    edge_index = edge_index)
            theta = 0.4
            v1 = x1[1].detach().cpu()
            v1_rot = rotate(theta,v1)

            v2 = rotate(theta, v)
            pos2 = rotate(theta, pos)
            y_force2 = rotate(theta, y_force)
            boundary_norm2 = rotate(theta, boundary_norm)
            x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force2, 
                                    pos = pos2, 
                                    edge_index = edge_index)
            v2_rot = x2[1]
            print(torch.norm(v1_rot - v2_rot)) 


class SO2MessagePassing(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32, rot_feed_forward = False, updated_node_embedding = False):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqMLPBlock(self.r_dim, n_scalars,L_max, num_rep, rot_feed_forward)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            
            pos = pos -pos.mean(dim = 0)
            x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force, 
                                    pos = pos, 
                                    edge_index = edge_index)
            theta = 0.4
            v1 = x1[1].detach().cpu()
            v1_rot = rotate(theta,v1)

            v2 = rotate(theta, v)
            pos2 = rotate(theta, pos)
            y_force2 = rotate(theta, y_force)
            boundary_norm2 = rotate(theta, boundary_norm)
            x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force2, 
                                    pos = pos2, 
                                    edge_index = edge_index)
            v2_rot = x2[1]
            print(torch.norm(v1_rot - v2_rot)) 


#import time
class SO2MessagePassingNodeAttr(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32, rot_feed_forward = False, updated_node_embedding = False):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqMLPNodeAttrBlock(self.r_dim, n_scalars,L_max, num_rep,n_scalars_node_attr = time_slize, num_rep_node_attr = time_slize + 2, rot_feed_forward = rot_feed_forward)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        
        node_attr_rot = torch.cat([v, y_force.unsqueeze(1), boundary_norm.unsqueeze(1)], dim = 1)
        node_attr_scalar = u
        node_attr = {"scalar":node_attr_scalar, "rot":  node_attr_rot}
        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        print(x['rot'].shape)
        for block in self.blocks:
            x = block( x, node_attr, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            
            pos = pos -pos.mean(dim = 0)
            x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force, 
                                    pos = pos, 
                                    edge_index = edge_index)
            theta = 0.4
            v1 = x1[1].detach().cpu()
            v1_rot = rotate(theta,v1)

            v2 = rotate(theta, v)
            pos2 = rotate(theta, pos)
            y_force2 = rotate(theta, y_force)
            boundary_norm2 = rotate(theta, boundary_norm)
            x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force2, 
                                    pos = pos2, 
                                    edge_index = edge_index)
            v2_rot = x2[1]
            print(torch.norm(v1_rot - v2_rot)) 

   


class InvariantTransformer(nn.Module):
    def __init__(self, time_slize, r_dim = 16, num_layers = 5, n_scalars = 32):
        super().__init__()  

        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = Node_embedding_inv(time_slize,n_scalars)
        blocks = []

        for i in range(num_layers):
            blocks.append(InvBlock(self.r_dim, n_scalars)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = torch.nn.LayerNorm(n_scalars)
        self.out_rot = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  2)
        self.out_scalar = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  1)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding)
        x = self.layer_norm(x)
        
        v_ = self.out_rot(x)
        u_ = self.out_scalar(x)
        return u[:,-1] + u_.view(-1), v_ + v[:,-1,:]
    
class InvariantMessagePassing(nn.Module):
    def __init__(self, time_slize, r_dim = 16, num_layers = 5, n_scalars = 32):
        super().__init__()  

        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = Node_embedding_inv(time_slize,n_scalars)
        blocks = []

        for i in range(num_layers):
            blocks.append(InvMPBlock(self.r_dim, n_scalars)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = torch.nn.LayerNorm(n_scalars)
        self.out_rot = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  2)
        self.out_scalar = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  1)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding)
        x = self.layer_norm(x)
        
        v_ = self.out_rot(x)
        u_ = self.out_scalar(x)
        return u[:,-1] + u_.view(-1), v_ + v[:,-1,:]
    


class SO2TransformerSmoke(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = Node_embedding_smoke(time_slize,n_scalars, L_max, num_rep)

        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlock(self.r_dim, n_scalars,L_max, num_rep, True)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,is_inflow,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, is_inflow,rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            
            pos = pos -pos.mean(dim = 0)
            x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force, 
                                    pos = pos, 
                                    edge_index = edge_index)
            theta = 0.4
            v1 = x1[1].detach().cpu()
            v1_rot = rotate(theta,v1)

            v2 = rotate(theta, v)
            pos2 = rotate(theta, pos)
            y_force2 = rotate(theta, y_force)
            boundary_norm2 = rotate(theta, boundary_norm)
            x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                    is_boundary = is_boundary, 
                                    y_force = y_force2, 
                                    pos = pos2, 
                                    edge_index = edge_index)
            v2_rot = x2[1]
            print(torch.norm(v1_rot - v2_rot))     



class InvariantTransformerSmoke(nn.Module):
    def __init__(self, time_slize, r_dim = 16, num_layers = 5, n_scalars = 32):
        super().__init__()  

        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = Node_embedding_inv_smoke(time_slize,n_scalars)
        blocks = []

        for i in range(num_layers):
            blocks.append(InvBlock(self.r_dim, n_scalars)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = torch.nn.LayerNorm(n_scalars)
        self.out_rot = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  2)
        self.out_scalar = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  1)

    def forward(self,u, v, boundary_norm, is_boundary,is_inflow, y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, is_inflow, y_force)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding)
        x = self.layer_norm(x)
        
        v_ = self.out_rot(x)
        u_ = self.out_scalar(x)
        return u[:,-1] + u_.view(-1), v_ + v[:,-1,:]
    

class RotTetrisModel(nn.Module):
    def __init__(self, num_rep = 16, r_dim = 16, num_layers = 3, n_scalars = 16, n_classes = 6,
                 N = 16, pointwise = False):
        super().__init__()  

        self.L_max = 1
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        self.edge_embedding = Edge_embedding_tetris(n_scalars, num_rep)

        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlock(self.r_dim, n_scalars,1, num_rep, True, N = N, pointwise = pointwise)) 

        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, self.L_max, num_rep)
        self.out_scalar = SO2OutU(self.L_max,num_rep, n_scalars,n_outputs=n_classes)

    def forward(self,pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)

        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.edge_embedding(edge_vec, rot,edge_index)

        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)        
        u_out = self.out_scalar(x, rot_theta)
        return scatter(u_out, batch, dim=0)
    
class InvTetrisModel(nn.Module):
    def __init__(self, r_dim = 16, num_layers = 3, n_scalars = 32, n_classes = 6):
        super().__init__()  

        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = Edge_embedding_tetris_inv(n_scalars)
        blocks = []

        for i in range(num_layers):
            blocks.append(InvMPBlock(self.r_dim, n_scalars)) 
        self.blocks = nn.ModuleList(blocks)
        self.out_scalar = MLP(in_dim = n_scalars,hidden_list = [n_scalars*3], out_dim =  n_classes)

    def forward(self,pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(edge_vec, edge_index = edge_index)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding)
        
        u_ = self.out_scalar(x)
        return scatter(u_, batch, dim=0)
    



class SO2TransformerTest(nn.Module):
    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 2, n_scalars = 32, 
                 rot_feed_forward = False, updated_node_embedding = False,N = 128, pointwise = False):
        super().__init__()  

        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlock(self.r_dim, n_scalars,L_max, num_rep, rot_feed_forward, N = N, pointwise = pointwise)) 
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
    
    def test_equivariance_model(self, u,v,  boundary_norm, 
                            is_boundary, 
                            y_force, 
                            pos, 
                            edge_index):
            errors = []
            times = []
            import time
            for i in range(50):
                start = time.time()
                pos = pos -pos.mean(dim = 0)
                x1 = self.forward(u = u,v = v,  boundary_norm = boundary_norm, 
                                        is_boundary = is_boundary, 
                                        y_force = y_force, 
                                        pos = pos, 
                                        edge_index = edge_index)
                times.append(time.time()-start)
                theta = torch.randn(1).item()
                v1 = x1[1].detach().cpu()
                v1_rot = rotate(theta,v1)

                v2 = rotate(theta, v)
                pos2 = rotate(theta, pos)
                y_force2 = rotate(theta, y_force)
                boundary_norm2 = rotate(theta, boundary_norm)
                x2 =self.forward(u = u,v = v2,  boundary_norm = boundary_norm2, 
                                        is_boundary = is_boundary, 
                                        y_force = y_force2, 
                                        pos = pos2, 
                                        edge_index = edge_index)
                v2_rot = x2[1]
                errors.append(torch.norm(v1_rot - v2_rot).item())
            return errors, times
    

class RotTetrisModelSimple(nn.Module):
    def __init__(self, num_rep = 16, r_dim = 16, num_layers = 3, n_scalars = 16, n_classes = 6,
                 N = 16, pointwise = False, rot_feed_forward = True):
        super().__init__()  

        self.L_max = 4
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        self.edge_embedding = Edge_embedding_tetris(n_scalars, num_rep)

        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlockSimple(self.r_dim, n_scalars,self.L_max , num_rep, rot_feed_forward, N = N, pointwise = pointwise)) 

        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, self.L_max, num_rep)
        self.out_scalar = SO2OutU(self.L_max,num_rep, n_scalars,n_outputs=n_classes)

    def forward(self,pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)

        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.edge_embedding(edge_vec, rot,edge_index)

        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)        
        v_out = self.out_rot(x, rot_theta)

        return scatter(u_out, batch, dim=0)
    

from .blocks import EqBlockSimple2

class RotPendulumModelSimple(nn.Module):
    def __init__(self, num_rep = 16, r_dim = 16, num_layers = 3, n_scalars = 16, escnn_feed_forward = False,escnn_conv= False):
        super().__init__()  

        self.L_max = 1
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        self.node_embedding = self.node_embedding = Node_embedding_pendulum(n_scalars, self.L_max, num_rep)

        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlockSimple2(self.r_dim, n_scalars , num_rep, escnn_feed_forward,escnn_conv)) 

        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, self.L_max, num_rep)
        self.out_rot = SO2OutV(self.L_max,num_rep, n_scalars)

    def forward(self,v,force, is_moving, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)

        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(v, force, is_moving, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)      
        out = self.out_rot(x, rot_theta)
        return out
    


class RotNavierModelSimple(nn.Module):

    def __init__(self, time_slize, L_max = 7, num_rep = 16, r_dim = 16, num_layers = 5, n_scalars = 32, escnn_feed_forward = False,escnn_conv = False,rot_feed_forward = False, updated_node_embedding = False, device = "cuda"):
        super().__init__()  

        L_max = 1
        self.L_max = L_max
        self.num_rep = num_rep
        self.r_dim = r_dim
        self.n_layers = num_layers
        if updated_node_embedding:
            self.node_embedding = Node_embedding_updated2(time_slize,n_scalars, L_max, num_rep)

        else:
            self.node_embedding = Node_embedding_vel(time_slize,n_scalars, L_max, num_rep)
        blocks = []

        for i in range(num_layers):
            blocks.append(EqBlockSimple2(self.r_dim, n_scalars , num_rep,  escnn_feed_forward = escnn_feed_forward,escnn_conv = escnn_conv, device = device) )
        self.blocks = nn.ModuleList(blocks)
        self.layer_norm = SO2LayerNorm(n_scalars, L_max, num_rep)
        self.out_rot = SO2OutV(L_max,num_rep, n_scalars)
        self.out_scalar = SO2OutU(L_max,num_rep, n_scalars)

    def forward(self,u, v, boundary_norm, is_boundary,y_force, pos, edge_index, batch = None):
        

        with torch.no_grad():
            if batch is None:
                pos = pos - pos.mean(dim = 0)
            else:
                mean = scatter(pos, batch, dim=0, reduce ='mean') 
                pos = pos - mean[batch]
            rot_theta = get_rot(pos, self.L_max)

            row, col = edge_index
            edge_vec = pos[row]- pos[col] 
            rot = get_rot(edge_vec, self.L_max)
            distance = torch.norm(edge_vec, dim = 1)
        dist_embedding = besel_linspace(distance,start = 0, end = 1, number = self.r_dim, cutoff = False)
        x = self.node_embedding(u, v, boundary_norm, y_force, is_boundary, rot_theta)
        for block in self.blocks:
            x = block( x, edge_index,dist_embedding, rot, rot_theta)
        x = self.layer_norm(x)
        
        v_out = self.out_rot(x, rot_theta)
        u_out = self.out_scalar(x, rot_theta)
        return u[:,-1] + u_out.view(-1), v_out + v[:,-1,:]
        
