import torch
import torch.nn.functional as F
from .util import get_rot_mulitK
class FourierPointwise(torch.nn.Module):
    
    def __init__(
            self,
            irr_max: int,
            function: str = 'p_relu',
            inplace: bool = True,
            N =128
    ):

        super(FourierPointwise, self).__init__()
        if function == 'p_relu':
            self._function = F.relu_ if inplace else F.relu
        elif function == 'p_elu':
            self._function = F.elu_ if inplace else F.elu
        elif function == 'p_sigmoid':
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == 'p_tanh':
            self._function = torch.tanh_ if inplace else F.tanh
        elif function == "p_silu":
            self._function = torch.nn.SiLU() if inplace else torch.nn.SiLU(inplace=True)
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))
        self.irr_size = irr_max*2

        with torch.no_grad():
            grid2 = (torch.linspace(0,2*torch.pi,N + 1)[0:-1]).view(-1,1)
            cos = torch.cos(grid2*torch.arange(1,irr_max +1).view(1,-1))
            sin = torch.sin(grid2*torch.arange(1,irr_max +1).view(1,-1))
            A = torch.stack((cos,sin), dim=2).view(N,self.irr_size)

            A_out = A.clone().detach().requires_grad_(True)
            eps = 1e-9
            Ainv = torch.linalg.inv(A_out.T @ A_out + eps * torch.eye(irr_max*2)) @ A_out.T

        self.register_buffer('A', A)
        self.register_buffer('Ainv', Ainv)
        self.irr_max = irr_max
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_hat = input#.view(shape[0], 1, self.irr_size, *shape[2:])
        x = torch.einsum('bcf...,gf->bcg...', x_hat, self.A)
        y = self._function(x)
        y_hat = torch.einsum('bcg...,fg->bcf...', y, self.Ainv)
        return y_hat
    
    def test_equivariance(self):
        theta = 0.83
        n_rep = 7
        rot = get_rot_mulitK(theta, self.irr_max)
        x = torch.randn(12, n_rep, self.irr_max*2)
        size = x.shape[:-1] + (self.irr_max,2)
        x_rot=torch.einsum("njkm, nkml -> njkl", x.view(size), rot).flatten(start_dim=2)
        out = self.forward(x)
        out_rot = self.forward(x_rot)
        out_rot_2 = torch.einsum("njkm, nkml -> njkl", out.view(size), rot).flatten(start_dim=2)

        print(torch.norm(out_rot_2-out_rot))



class RotActivation(torch.nn.Module):
    
    def __init__(
            self,
            L_max: int,
            function: str = 'p_relu',
            inplace: bool = True,
    ):

        super(RotActivation, self).__init__()
        if function == 'p_relu':
            self._function = F.relu_ if inplace else F.relu
        elif function == 'p_elu':
            self._function = F.elu_ if inplace else F.elu
        elif function == 'p_sigmoid':
            self._function = torch.sigmoid_ if inplace else F.sigmoid
        elif function == 'p_tanh':
            self._function = torch.tanh_ if inplace else F.tanh
        elif function == "p_silu":
            self._function = torch.nn.SiLU() if inplace else torch.nn.SiLU(inplace=True)
        elif function == "p_leaky_relu":
            self._function = torch.nn.LeakyReLU() if inplace else torch.nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Function "{}" not recognized!'.format(function))

        self.L_max = L_max
        
        
    def forward(self, input: torch.Tensor, rot = None, pos = None, ) -> torch.Tensor:

        if rot is None:
            rot = get_rot(pos,self.L_max)
        rot_inv = torch.transpose(rot, 2,3).contiguous()

        size = input.shape[:-1] + (self.L_max,2)    

        x=torch.einsum("njkm, nkml -> njkl", input.view(size), rot_inv).flatten(start_dim=2)        
        y = self._function(x)
        y_hat = torch.einsum("njkm, nkml -> njkl", y.view(size), rot).flatten(start_dim=2) 
        return y_hat
    
    def test_equivariance(self, theta = 0.83):

        n_rep =1
        pos = torch.randn(10,2)
        pos = pos - pos.mean(dim = 0)
        input = torch.randn(10,n_rep,self.L_max*2)
        size = input.shape[:-1] + (self.L_max,2)

        rot = get_rot_mulitK(theta, self.L_max)

        out1 = self.forward(input, pos = pos)
        out1_rot = torch.einsum("njkm, nkml -> njkl", out1.view(size), rot).flatten(start_dim=2)
        
        input_rot=torch.einsum("njkm, nkml -> njkl", input.view(size), rot).flatten(start_dim=2)
        pos_rot = pos@rot[0,0,:,:]
        out2_rot = self.forward(input_rot, pos = pos_rot)
        print(torch.norm(out2_rot-out1_rot))



def get_rot(edge_vec, L_max):
    theta = torch.atan2(edge_vec[:,1],edge_vec[:,0]).reshape(-1,1)
    k = torch.arange(1,L_max + 1).reshape(1,-1).to(edge_vec.device) 
    k  = theta*k
    s = torch.sin(k)
    c = torch.cos(k)
    A1 = torch.stack((c, -s)).permute(1,2,0)
    A2 = torch.stack((s, c)).permute(1,2,0)
    A_perm = torch.stack((A1, A2)).permute(1,2,3,0)
    return A_perm