from torch import tensor
import torch

m = n = 1
R = tensor([1.0])
Q = tensor([1.0])

x0 = tensor([5.0])
p0 = tensor([10.0])

a = 5
b = -1
k=0
F = torch.tensor(0.9)

def motion_model(x, Q, noise=False):
    w_k = 0
    global k
    k = k + 1
    x = x.view([-1])
    if  noise: w_k = torch.randn(x.size(0))
    return 0.9*x + torch.sqrt(Q) * w_k
    # return 0.5*x + 25*x / (1+x.pow(2)) +\
    #        8*torch.cos(tensor(1.2*(k-1)))+ torch.sqrt(Q) * w_k

def measurement_model(x, a, b, R, noise=False):
    v_k = 0
    x = x.view([-1])
    if noise: v_k = torch.randn(x.size(0))
    return a*x + b*x.pow(3) + torch.sqrt(R) * v_k
    # return x.pow(2) / 20 + torch.sqrt(R) * v_k