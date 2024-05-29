
import torch
from torch import nn
import math

def rotate(theta, x_):
    M = torch.tensor([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
    return x_@M

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

def get_rot_mulitK(theta, L_max):
    k = torch.arange(1,L_max + 1).reshape(1,-1)
    k  = theta*k
    s = torch.sin(k)
    c = torch.cos(k)
    A1 = torch.stack((c, s)).permute(1,2,0)
    A2 = torch.stack((-s, c)).permute(1,2,0)
    A_perm = torch.stack((A1, A2)).permute(1,2,3,0)
    return A_perm

def besel_linspace(x: torch.Tensor, start, end, number, cutoff=None) -> torch.Tensor:

    # pylint: disable=misplaced-comparison-constant

    if cutoff not in [True, False]:
        raise ValueError("cutoff must be specified")

    if not cutoff:
        values = torch.linspace(start, end, number, dtype=x.dtype, device=x.device)
    else:
        values = torch.linspace(start, end, number + 2, dtype=x.dtype, device=x.device)
        values = values[1:-1]


    x = x[..., None] - start
    c = end - start
    bessel_roots = torch.arange(1, number + 1, dtype=x.dtype, device=x.device) * math.pi
    out = math.sqrt(2 / c) * torch.sin(bessel_roots * x / c) / x

    if not cutoff:
        return out
    else:
        return out * ((x / c) < 1) * (0 < x)
        
activations = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'gelu': nn.GELU(),
    'leaky_relu': nn.LeakyReLU(),
}

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_list, out_dim, activation='relu'):
        
        super().__init__()
        assert activation in ['relu', 'tanh', 'gelu','leaky_relu']
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_list[0]))
        self.layers.append(activations[activation])
        
        for i in range(len(hidden_list)-1):
            self.layers.append(nn.Linear(hidden_list[i], hidden_list[i+1]))
            self.layers.append(activations[activation])
        self.layers.append(nn.Linear(hidden_list[-1],out_dim))
    def reset_parameters(self):
        for l in self.layers:
            if isinstance(l, torch.nn.Linear):
                l.reset_parameters()
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out