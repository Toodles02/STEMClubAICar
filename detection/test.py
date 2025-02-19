import torch
import numpy as np 

dim1_tensor = torch.randn(3) 
print(dim1_tensor.shape)
print(torch.dim(dim1_tensor, 0))