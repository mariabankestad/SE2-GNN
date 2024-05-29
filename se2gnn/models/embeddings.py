
import torch
from torch import nn
from torch_geometric.utils import softmax, scatter
from .util import MLP

from .so2layers import SO2MLP

class Node_edge_embedding(nn.Module):
    def __init__(self, r_dim, u_dim, n_scalars, L_max, num_rep):
        super().__init__()
        dim = r_dim + u_dim
        self.edge_embedding = MLP(in_dim = dim,hidden_list = [dim*4]*2, out_dim =  L_max*num_rep*2)
        self.node_embedding = MLP(in_dim = u_dim,hidden_list = [u_dim*4]*2, out_dim =  n_scalars)
        self.L_max = L_max
        self.num_rep = num_rep

        self.act_alpha = nn.LeakyReLU(inplace=True)

        self.linear_alpha = nn.Linear(L_max*num_rep*2,1)
        self.layer_norm = nn.LayerNorm(L_max*num_rep*2)


    def forward(self, dist_embedding, u, rot, edge_index):
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        row, col = edge_index
        du = u.clone()
        du = du[row] - du[col]
        dist_embedding.device
        edge_embedding = self.edge_embedding(torch.cat([dist_embedding, du], dim = 1))


        alpha = self.act_alpha(self.layer_norm(edge_embedding))
        alpha = self.linear_alpha(alpha)
        alpha = softmax(alpha, col, num_nodes=u.size(0)).reshape(-1,1)

        edge_embedding = edge_embedding.reshape(-1, self.num_rep, self.L_max*2)
        edge_embedding = edge_embedding*alpha.unsqueeze(-1)
        edge_embedding = torch.einsum("njkm, nkml -> njkl", edge_embedding.view(edge_embedding.shape[:-1] + (self.L_max,2)), rot_inv).flatten(start_dim=2)
        
        out_edge = scatter(edge_embedding, col, dim = 0, dim_size=u.size(0))
        out_node = self.node_embedding(u)

        node_embedding = {"scalar": out_node, "rot": out_edge}
        return node_embedding
    


class Scalar_embedding(torch.nn.Module):
    def __init__(self,n_scalars, L_max, num_rep_in, output_dim):
        super().__init__()

        input_dim = n_scalars + L_max*2*num_rep_in
        self.L_max = L_max
        self.mlp = MLP(in_dim = input_dim,hidden_list = [output_dim*3], out_dim =  output_dim)

    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x_rot.shape[:-1] + (self.L_max,2)
        x_rot = torch.einsum("njkm, nkml -> njkl",x_rot.view(size), rot_theta_inv).flatten(start_dim=1)
        x = torch.cat([x_scalars,x_rot], dim = 1)
        x = self.mlp(x)
        return x
    
class Rot_embedding(torch.nn.Module):
    def __init__(self,n_scalars, L_max, num_rep_in, num_rep_out):
        super().__init__()

        input_dim = n_scalars + L_max*2*num_rep_in
        self.L_max = L_max
        self.num_rep_out = num_rep_out
        self.output_dim = num_rep_out*L_max*2
        self.mlp = MLP(in_dim = input_dim,hidden_list = [self.output_dim*3], out_dim =  self.output_dim)

    def forward(self, x, rot_theta):
        x_scalars = x['scalar']
        x_rot = x['rot']
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x_rot.shape[:-1] + (self.L_max,2)
        x_rot = torch.einsum("njkm, nkml -> njkl",x_rot.view(size), rot_theta_inv).flatten(start_dim=1)
        x = torch.cat([x_scalars,x_rot], dim = 1)
        x = self.mlp(x)
        x = x.reshape(size[0], self.num_rep_out, self.L_max*2)
        x = torch.einsum("njkm, nkml -> njkl", x.view(x.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim=2)
        return x

class Rot_embedding2(torch.nn.Module):
    def __init__(self,L_max, num_rep_in, num_rep_out):
        super().__init__()

        input_dim = L_max*2*num_rep_in
        self.L_max = L_max
        self.num_rep_out = num_rep_out
        self.output_dim = num_rep_out*L_max*2
        self.mlp = MLP(in_dim = input_dim,hidden_list = [self.output_dim*3], out_dim =  self.output_dim)

    def forward(self, x, rot_theta):
        rot_theta_inv = torch.transpose(rot_theta, 2,3).contiguous()
        size = x.shape[:-1] + (self.L_max,2)
        x = torch.einsum("njkm, nkml -> njkl",x.view(size), rot_theta_inv).flatten(start_dim=1)
        x = self.mlp(x)
        x = x.reshape(size[0], self.num_rep_out, self.L_max*2)
        x = torch.einsum("njkm, nkml -> njkl", x.view(x.shape[:-1] + (self.L_max,2)), rot_theta).flatten(start_dim=2)
        return x
    

class Node_embedding_vel(torch.nn.Module):
    def __init__(self, n_traj, n_scalars, L_max, num_rep):
        super().__init__()

        self.scalar_layer = MLP(in_dim = n_traj*2 + 1,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.L_max = L_max
        self.num_rep = num_rep
        self.rot_layer = SO2MLP(L_max=L_max,num_rep_in=n_traj+2,num_rep_hidden=num_rep*3, num_rep_out=num_rep)
        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, force, is_boundary, rot_theta):
        N = u.shape[0]
        v_norm =torch.norm(v, dim = 2)
        force_norm = torch.norm(force, dim = 1)
        v_hat = (v/v_norm.unsqueeze(-1))
        f_hat = (force/force_norm.unsqueeze(-1))

        theta_v = torch.atan2(v_hat[:,:,1],v_hat[:,:,0])
        theta_n = torch.atan2(boundary_norm[is_boundary,1],boundary_norm[is_boundary,0]).reshape(-1,1)
        theta_f = torch.atan2(f_hat[:,1],f_hat[:,0]).reshape(-1,1)

        k = torch.arange(1,self.L_max+ 1).reshape(1,-1).to(u.device) 
        k_v = theta_v[:,:,None]*k[None,:]
        k_n  = theta_n*k
        k_f  = theta_f*k

        sv = torch.sin(k_v); cv = torch.cos(k_v)
        sn = torch.sin(k_n); cn = torch.cos(k_n)
        sf = torch.sin(k_f); cf = torch.cos(k_f)
        
        rot_features_v = torch.stack((cv,sv), dim=3).view(N,self.n_traj,self.L_max*2)
        rot_features_f = torch.stack((cf,sf), dim=2).view(N,self.L_max*2)
        rot_features_b_ = torch.stack((cn,sn), dim=2).view(theta_n.shape[0],self.L_max*2)
        rot_features_b = torch.zeros(N,self.L_max*2).to(u.device)
        rot_features_b[is_boundary,:] = rot_features_b_

        scalar_features = torch.cat([u, v_norm, force_norm.view(-1,1)], dim = 1)
        scalar_features = self.scalar_layer(scalar_features)
        

        rot_featuers = torch.cat([rot_features_b[:,None,:], rot_features_v, rot_features_f[:,None,:]], dim = 1)
        rot_featuers = self.rot_layer(rot_featuers, rot_theta)

        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding
    
class Node_embedding_updated(torch.nn.Module):
    def __init__(self, n_traj, n_scalars, L_max, num_rep):
        super().__init__()

        self.scalar_layer = Scalar_embedding(n_scalars = n_traj*2 + 1, L_max = L_max, num_rep_in=n_traj+2, output_dim=n_scalars)

        self.L_max = L_max
        self.num_rep = num_rep
        self.rot_layer = Rot_embedding(n_scalars=n_traj*2 + 1, L_max=L_max, num_rep_in=n_traj+2, num_rep_out=num_rep)

        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, force, is_boundary, rot_theta):
        N = u.shape[0]
        v_norm =torch.norm(v, dim = 2)
        force_norm = torch.norm(force, dim = 1)
        v_hat = (v/v_norm.unsqueeze(-1))
        f_hat = (force/force_norm.unsqueeze(-1))

        theta_v = torch.atan2(v_hat[:,:,1],v_hat[:,:,0])
        theta_n = torch.atan2(boundary_norm[is_boundary,1],boundary_norm[is_boundary,0]).reshape(-1,1)
        theta_f = torch.atan2(f_hat[:,1],f_hat[:,0]).reshape(-1,1)

        k = torch.arange(1,self.L_max+ 1).reshape(1,-1).to(u.device) 
        k_v = theta_v[:,:,None]*k[None,:]
        k_n  = theta_n*k
        k_f  = theta_f*k

        sv = torch.sin(k_v); cv = torch.cos(k_v)
        sn = torch.sin(k_n); cn = torch.cos(k_n)
        sf = torch.sin(k_f); cf = torch.cos(k_f)
        
        rot_features_v = torch.stack((cv,sv), dim=3).view(N,self.n_traj,self.L_max*2)
        rot_features_f = torch.stack((cf,sf), dim=2).view(N,self.L_max*2)
        rot_features_b_ = torch.stack((cn,sn), dim=2).view(theta_n.shape[0],self.L_max*2)
        rot_features_b = torch.zeros(N,self.L_max*2).to(u.device)
        rot_features_b[is_boundary,:] = rot_features_b_

        scalar_features = torch.cat([u, v_norm, force_norm.view(-1,1)], dim = 1)
        rot_featuers = torch.cat([rot_features_b[:,None,:], rot_features_v, rot_features_f[:,None,:]], dim = 1)
        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        scalar_features = self.scalar_layer(node_embedding, rot_theta)
        rot_featuers = self.rot_layer(node_embedding, rot_theta)

        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding

class Node_embedding_updated2(torch.nn.Module):
    def __init__(self, n_traj, n_scalars, L_max, num_rep):
        super().__init__()

        self.scalar_layer = MLP(in_dim = n_traj*2 + 1,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.L_max = L_max
        self.num_rep = num_rep
        self.rot_layer = Rot_embedding2(L_max=L_max, num_rep_in=n_traj+2, num_rep_out=num_rep)

        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, force, is_boundary, rot_theta):
        N = u.shape[0]
        v_norm =torch.norm(v, dim = 2)
        force_norm = torch.norm(force, dim = 1)
        v_hat = (v/v_norm.unsqueeze(-1))
        f_hat = (force/force_norm.unsqueeze(-1))

        theta_v = torch.atan2(v_hat[:,:,1],v_hat[:,:,0])
        theta_n = torch.atan2(boundary_norm[is_boundary,1],boundary_norm[is_boundary,0]).reshape(-1,1)
        theta_f = torch.atan2(f_hat[:,1],f_hat[:,0]).reshape(-1,1)

        k = torch.arange(1,self.L_max+ 1).reshape(1,-1).to(u.device) 
        k_v = theta_v[:,:,None]*k[None,:]
        k_n  = theta_n*k
        k_f  = theta_f*k

        sv = torch.sin(k_v); cv = torch.cos(k_v)
        sn = torch.sin(k_n); cn = torch.cos(k_n)
        sf = torch.sin(k_f); cf = torch.cos(k_f)
        
        rot_features_v = torch.stack((cv,sv), dim=3).view(N,self.n_traj,self.L_max*2)
        rot_features_f = torch.stack((cf,sf), dim=2).view(N,self.L_max*2)
        rot_features_b_ = torch.stack((cn,sn), dim=2).view(theta_n.shape[0],self.L_max*2)
        rot_features_b = torch.zeros(N,self.L_max*2).to(u.device)
        rot_features_b[is_boundary,:] = rot_features_b_

        scalar_features = torch.cat([u, v_norm, force_norm.view(-1,1)], dim = 1)
        rot_featuers = torch.cat([rot_features_b[:,None,:], rot_features_v, rot_features_f[:,None,:]], dim = 1)
        #node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        scalar_features = self.scalar_layer(scalar_features)
        rot_featuers = self.rot_layer(rot_featuers, rot_theta)

        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding



class Edge_embedding_tetris(torch.nn.Module):
    def __init__(self, n_scalars, num_rep):
        super().__init__()

        self.scalar_layer = MLP(in_dim = 1,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.L_max = 1
        self.num_rep = num_rep
        self.rot_layer = Rot_embedding2(L_max=self.L_max, num_rep_in=1, num_rep_out=num_rep)
        self.message_function =  MLP(in_dim = 1,hidden_list = [num_rep*2*3], out_dim =  num_rep*2)

    def forward(self, v, rot, edge_index):
        row,col = edge_index
        N = v.shape[0]
        v_norm =torch.norm(v, dim = 1).unsqueeze(-1)
        rot_inv = torch.transpose(rot, 2,3).contiguous()
        x_out = self.message_function(v_norm)
        mess_rot = x_out.reshape(x_out.size(0), self.num_rep, 2)
        mess_rot = torch.einsum("njkm, nkml -> njkl", mess_rot.view(mess_rot.shape[:-1] + (1,2)), rot).flatten(start_dim=2)
        rot_featuers = scatter(mess_rot, col, dim = 0, dim_size=edge_index.max()+1)
        scalar_features = self.scalar_layer(v_norm)
        scalar_features = scatter(scalar_features, col, dim = 0, dim_size=edge_index.max()+1)
        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding

class Node_embedding_inv(torch.nn.Module):
    def __init__(self, n_traj, n_scalars):
        super().__init__()
        dim = n_traj + n_traj*2 + 4
        self.scalar_layer = MLP(in_dim = dim,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, force):

        v = v.flatten(start_dim=1)
        scalar_features = torch.cat([u, v, force, boundary_norm], dim = 1)
        scalar_features = self.scalar_layer(scalar_features)
        return scalar_features
    
class Node_embedding_pendulum(torch.nn.Module):
    def __init__(self, n_scalars, L_max, num_rep):
        super().__init__()

        self.scalar_layer = MLP(in_dim = 4,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.L_max = L_max
        self.num_rep = num_rep
        self.rot_layer = Rot_embedding2(L_max=L_max, num_rep_in=2, num_rep_out=num_rep)


    def forward(self, v, force,is_moving, rot_theta):
        N = force.shape[0]
        force_norm = torch.norm(force, dim = 1)
        f_hat = (force/force_norm.unsqueeze(-1))

        v_norm = torch.norm(v, dim = 1)
        v_hat = (v/v_norm.unsqueeze(-1))

        theta_f = torch.atan2(f_hat[:,1],f_hat[:,0]).reshape(-1,1)
        theta_v = torch.atan2(v_hat[:,1],v_hat[:,0]).reshape(-1,1)

        is_moving = torch.nn.functional.one_hot(is_moving.long())
        k = torch.arange(1,self.L_max+ 1).reshape(1,-1).to(force.device) 
        k_f  = theta_f*k
        k_v  = theta_v*k

        sf = torch.sin(k_f); cf = torch.cos(k_f)
        sv = torch.sin(k_v); cv = torch.cos(k_v)

        
        rot_features_f = torch.stack((cf,sf), dim=2).view(N,self.L_max*2)
        rot_features_v = torch.stack((cv,sv), dim=2).view(N,self.L_max*2)


        scalar_features = torch.cat([force_norm.view(-1,1),v_norm.view(-1,1),is_moving], dim = 1) 

        rot_featuers = torch.cat([rot_features_f[:,None,:],rot_features_v[:,None,:]], dim = 1)
        #node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        scalar_features = self.scalar_layer(scalar_features)
        rot_featuers = self.rot_layer(rot_featuers, rot_theta)

        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding

class Node_embedding_smoke(torch.nn.Module):
    def __init__(self, n_traj, n_scalars, L_max, num_rep):
        super().__init__()

        self.scalar_layer = MLP(in_dim = n_traj*2 + 3,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.L_max = L_max
        self.num_rep = num_rep
        self.rot_layer = Rot_embedding2(L_max=L_max, num_rep_in=n_traj+2, num_rep_out=num_rep)

        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, force, is_boundary, is_inflow,rot_theta):
        N = u.shape[0]
        v_norm =torch.norm(v, dim = 2)
        force_norm = torch.norm(force, dim = 1)
        v_hat = (v/v_norm.unsqueeze(-1))
        f_hat = (force/force_norm.unsqueeze(-1))
        is_inflow = torch.nn.functional.one_hot(is_inflow.long())
        theta_v = torch.atan2(v_hat[:,:,1],v_hat[:,:,0])
        theta_n = torch.atan2(boundary_norm[is_boundary,1],boundary_norm[is_boundary,0]).reshape(-1,1)
        theta_f = torch.atan2(f_hat[:,1],f_hat[:,0]).reshape(-1,1)

        k = torch.arange(1,self.L_max+ 1).reshape(1,-1).to(u.device) 
        k_v = theta_v[:,:,None]*k[None,:]
        k_n  = theta_n*k
        k_f  = theta_f*k

        sv = torch.sin(k_v); cv = torch.cos(k_v)
        sn = torch.sin(k_n); cn = torch.cos(k_n)
        sf = torch.sin(k_f); cf = torch.cos(k_f)
        
        rot_features_v = torch.stack((cv,sv), dim=3).view(N,self.n_traj,self.L_max*2)
        rot_features_f = torch.stack((cf,sf), dim=2).view(N,self.L_max*2)
        rot_features_b_ = torch.stack((cn,sn), dim=2).view(theta_n.shape[0],self.L_max*2)
        rot_features_b = torch.zeros(N,self.L_max*2).to(u.device)
        rot_features_b[is_boundary,:] = rot_features_b_

        scalar_features = torch.cat([u, v_norm, force_norm.view(-1,1),is_inflow], dim = 1)
        rot_featuers = torch.cat([rot_features_b[:,None,:], rot_features_v, rot_features_f[:,None,:]], dim = 1)
        #node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        scalar_features = self.scalar_layer(scalar_features)
        rot_featuers = self.rot_layer(rot_featuers, rot_theta)

        node_embedding = {"scalar": scalar_features, "rot": rot_featuers}

        return node_embedding
    

class Node_embedding_inv_smoke(torch.nn.Module):
    def __init__(self, n_traj, n_scalars):
        super().__init__()
        dim = n_traj + n_traj*2 + 6
        self.scalar_layer = MLP(in_dim = dim,hidden_list = [n_scalars*3], out_dim =  n_scalars)
        self.n_traj = n_traj

    def forward(self, u, v, boundary_norm, is_inflow, force):
        is_inflow = torch.nn.functional.one_hot(is_inflow.long())

        v = v.flatten(start_dim=1)
        scalar_features = torch.cat([u, v, force,is_inflow, boundary_norm], dim = 1)
        scalar_features = self.scalar_layer(scalar_features)
        return scalar_features

class Edge_embedding_tetris_inv(torch.nn.Module):
    def __init__(self, n_scalars):
        super().__init__()
        dim = 2
        self.scalar_layer = MLP(in_dim = dim,hidden_list = [n_scalars*3], out_dim =  n_scalars)

    def forward(self, v,edge_index):
        row,col = edge_index
        scalar_features = self.scalar_layer(v)
        scalar_features = scatter(scalar_features, col, dim = 0, dim_size=edge_index.max()+1)
        return scalar_features
    