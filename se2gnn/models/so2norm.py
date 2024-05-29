
import torch
from torch import nn
from .util import get_rot_mulitK

class SO2LayerNorm(nn.Module):
    def __init__(self,n_scalars, L_max, num_rep):
        super().__init__()   

        self.affine_weight_l0 = nn.Parameter(torch.ones((1,n_scalars)))
        self.affine_weight_lplus = nn.Parameter(torch.ones(1,L_max*num_rep))
        self.affine_bias = nn.Parameter(torch.zeros(1,n_scalars))
        self.register_buffer('ind_select', torch.arange(0,L_max*num_rep).repeat_interleave(2))

        self.L_max = L_max
        self.num_rep = num_rep
        self.n_scalars = n_scalars

    def forward(self,x):
        x_scalars = x["scalar"]
        x_rot = x["rot"]
        if self.L_max > 0:
            x_rot_norm = x_rot.pow(2).mean(dim = (1,2), keepdim = True).sqrt()
            x_rot= x_rot/(x_rot_norm + 1e-6)
            weight = torch.index_select(self.affine_weight_lplus, dim=1, index=self.ind_select).reshape(self.num_rep,self.L_max*2)#.permute((0,2, 1)).reshape(1,-1)
            x_rot = x_rot* weight
        x_scalars = x_scalars- x_scalars.mean(dim = 1, keepdim=True)

        x_scalars_norm = x_scalars.pow(2).mean(dim = 1, keepdim=True).sqrt()


        x_scalars = x_scalars /(x_scalars_norm + 1e-6) 
        x_scalars = x_scalars*self.affine_weight_l0 +self.affine_bias
        return {"scalar": x_scalars, "rot": x_rot}
    
    def test_equivariance(self):
        theta = 0.83

        rot = get_rot_mulitK(theta, self.L_max)
        x_rot1 = torch.randn(12, self.num_rep, self.L_max*2)
        x_scalar = torch.randn(12, self.n_scalars)
        size = x_rot1.shape[:-1] + (self.L_max,2)
        x_rot2=torch.einsum("njkm, nkml -> njkl", x_rot1.view(size), rot).flatten(start_dim=2)
        input1 = {"scalar": x_scalar, "rot": x_rot1}
        input2 = {"scalar": x_scalar, "rot": x_rot2}
       
        out1 = self.forward(input1)
        out2 = self.forward(input2)
        out1_rot = out1['rot']
        out_rot_1 = torch.einsum("njkm, nkml -> njkl", out1_rot.view(size), rot).flatten(start_dim=2)
        out_rot_2 = out2['rot']
        print(torch.norm(out_rot_1-out_rot_2))