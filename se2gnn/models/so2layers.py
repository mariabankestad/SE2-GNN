
from .activation import FourierPointwise, RotActivation
from.util import get_rot_mulitK, get_rot,rotate
from torch import nn
import torch

class SO2Linear(nn.Module):
    def __init__(self, L_max,num_rep_in, num_rep_out):
        super().__init__()  
        
        linears = []
        for i in range(L_max):
            linear = torch.nn.Linear(num_rep_in,num_rep_out, bias = False)
            torch.nn.init.xavier_normal_(linear.weight)
            linears.append(linear)
        self.linears = nn.ModuleList(linears)

        self.num_rep_in = num_rep_in
        self.num_rep_out = num_rep_out

        self.L_max = L_max

    def forward(self, x):
        n = x.shape[0]
        x_ = x.reshape(n,self.num_rep_in, -1, 2).permute(0,2, 3, 1)
        outputs = []
        for i, layer in enumerate(self.linears):
            outputs.append(layer(x_[:,i]).transpose(-2,-1).contiguous())
        out = torch.cat(outputs, dim = 2)

        return out

from .util import MLP

  
class so2_linear_escnn_edge(torch.nn.Module):

  def __init__(self,
                irr_in,
                irr_out,
                n_scalar = None
                ):
    super().__init__()
    self.irr_in =  irr_in
    self.irr_out =  irr_out
    self.ir_1 = torch.cat([torch.tensor((1,0)) + i*2 for i in range(self.irr_in)])
    self.ir_2 = torch.ones(2*irr_in)
    self.ir_2[1::2] = -1
    dim = irr_in*irr_out*2
    self.weights_fc = MLP(n_scalar, [dim*2], dim)
  
  def get_weight_matrix(self, s_):

    w_ = self.weights_fc(s_).view(-1,self.irr_out,2*self.irr_in)

    w_1 = w_[:,:,self.ir_1.to(w_.device)]
    w_2 = w_*self.ir_2.to(w_.device)
    weight_matrix = (torch.stack((w_2, w_1), dim = 1)).permute(0,3,2,1).reshape(-1,self.irr_in*2,self.irr_out*2)
    self.weight_matrix = weight_matrix
    return weight_matrix
  

  def forward(self,x, s):
      x = x.flatten(start_dim = 1)
      w = self.get_weight_matrix(s)
      output = torch.bmm(x.unsqueeze(1), w)
      return output.view(-1,self.irr_out,2)
  def reset_parameters(self):
      self.weights_fc.reset_parameters()
  
class so2_linear_escnn(torch.nn.Module):

  def __init__(self,
                irr_in,
                irr_out,
                device = "cuda"
                ):
    super().__init__()
    self.irr_in =  irr_in
    self.irr_out =  irr_out
    self.ir_1 = torch.cat([torch.tensor((1,0)) + i*2 for i in range(self.irr_in)])
    self.ir_2 = torch.ones(2*irr_in)
    self.ir_2[1::2] = -1
    dim = irr_in*irr_out*2

    self.weights = torch.nn.Parameter(torch.zeros(dim, device = device), requires_grad=True).reshape(-1,1)
    torch.nn.init.xavier_normal_(self.weights)
  
  def get_weight_matrix(self):
    w_ = self.weights.view(self.irr_out,-1)
    w_1 = w_[:,self.ir_1.to(w_.device)]
    w_2 = w_*self.ir_2.to(w_.device)
    weight_matrix = (torch.stack((w_2, w_1), dim = 0).T).reshape(-1,self.irr_out*2).T
    return weight_matrix

  def forward(self,x):
      x = x.flatten(start_dim = 1)
      w = self.get_weight_matrix()
      output = torch.nn.functional.linear(x, w)
      return output.view(-1,self.irr_out,2)
  
class SO2MLP(nn.Module):
    def __init__(self, L_max,num_rep_in, num_rep_hidden, num_rep_out = None, pointwise = False, N = 128):
        super().__init__()  
        if num_rep_out == None:
            num_rep_out = num_rep_in
        self.pointwise = pointwise
        self.so2_linear1 = SO2Linear(L_max, num_rep_in,num_rep_hidden)
        if self.pointwise:
            self.act1 = FourierPointwise(irr_max=L_max, function="p_silu", N = N)
        else: 
            self.act1 = RotActivation(L_max=L_max, function="p_leaky_relu")
        self.L_max = L_max
        self.n_rep = num_rep_in
        
        self.so2_linear2 = SO2Linear(L_max, num_rep_hidden,num_rep_out)

    def forward(self, x, rot_theta = None):
        x = self.so2_linear1(x)
        if self.pointwise:
            x = self.act1(x) 
        else:
            x = self.act1(x, rot_theta)
        x = self.so2_linear2(x)
        return x  
    
    def test_equivariance(self,theta = 0.83):

        N = 12
        rot = get_rot_mulitK(theta, self.L_max)
        pos = torch.randn(N,2)
        pos = pos - pos.mean(dim = 0)
        input = torch.randn(N,self.n_rep,self.L_max*2)
        size = input.shape[:-1] + (self.L_max,2)

        rot_act1 = get_rot(pos,self.L_max)

        pos_rot = rotate(theta, pos)
        input_rot=torch.einsum("njkm, nkml -> njkl", input.view(size), rot).flatten(start_dim=2)
        rot_act2 = get_rot(pos_rot,self.L_max)

        out1 = self.forward(input, rot_act1)
        out1_rot = torch.einsum("njkm, nkml -> njkl", out1.view(size), rot).flatten(start_dim=2)
        out2 = self.forward(input_rot, rot_act2)
        print(torch.norm(out1_rot- out2))

class SO2MLP_escnn_mess(nn.Module):
    def __init__(self, num_rep_in, num_rep_hidden, n_scalar,num_rep_out = None, N = 64):
        super().__init__()  
        if num_rep_out == None:
            num_rep_out = num_rep_in
        self.so2_linear1 = so2_linear_escnn_edge(num_rep_in,num_rep_hidden,n_scalar= n_scalar)

        self.act1 = FourierPointwise(irr_max=1, function="p_silu", N = N)

        self.n_rep = num_rep_in
        self.n_scalar = n_scalar
        self.so2_linear2 = so2_linear_escnn_edge(num_rep_hidden,num_rep_out,n_scalar= n_scalar)

    def forward(self, x, s_):
        x = self.so2_linear1(x,s_)

        x = self.act1(x) 

        x = self.so2_linear2(x,s_)

        return x  
    
    def reset_parameters(self):
        self.so2_linear2.reset_parameters()
        self.so2_linear1.reset_parameters()

    
    def test_equivariance(self,theta = 0.83):

        N = 1
        rot = get_rot_mulitK(theta, 1)
        pos = torch.randn(N,2)
        pos = pos - pos.mean(dim = 0)
        scalars = torch.randn(N,self.n_scalar)
        input = torch.randn(N,self.n_rep,1*2)
        size = input.shape[:-1] + (1,2)
        input_rot=torch.einsum("njkm, nkml -> njkl", input.view(size), rot).flatten(start_dim=2)
        out1 = self.forward(input, scalars)
        size2 = out1.shape[:-1] + (1,2)
        out1_rot = torch.einsum("njkm, nkml -> njkl", out1.view(size2), rot).flatten(start_dim=2)
        out2 = self.forward(input_rot,scalars)
        print(torch.norm(out1_rot- out2))


class SO2MLP_escnn(nn.Module):
    def __init__(self, num_rep_in, num_rep_hidden,num_rep_out = None, N = 64, device = "cuda"):
        super().__init__()  
        if num_rep_out == None:
            num_rep_out = num_rep_in
        self.so2_linear1 = so2_linear_escnn(num_rep_in,num_rep_hidden, device = device)

        self.act1 = FourierPointwise(irr_max=1, function="p_silu", N = N)

        self.n_rep = num_rep_in
        self.so2_linear2 = so2_linear_escnn(num_rep_hidden,num_rep_out, device = device)

    def forward(self, x):
        x = self.so2_linear1(x)
        x = self.act1(x) 
        x = self.so2_linear2(x)
        return x  
    
    def test_equivariance(self,theta = 0.83):

        N = 12
        rot = get_rot_mulitK(theta, 1)
        pos = torch.randn(N,2)
        pos = pos - pos.mean(dim = 0)
        input = torch.randn(N,self.n_rep,1*2)
        size = input.shape[:-1] + (1,2)
        input_rot=torch.einsum("njkm, nkml -> njkl", input.view(size), rot).flatten(start_dim=2)
        out1 = self.forward(input)
        size2 = out1.shape[:-1] + (1,2)
        out1_rot = torch.einsum("njkm, nkml -> njkl", out1.view(size2), rot).flatten(start_dim=2)
        out2 = self.forward(input_rot)
        print(torch.norm(out1_rot- out2))