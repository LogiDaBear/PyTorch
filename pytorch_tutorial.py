import torch
###---BASIC PYTORCH---###


####TENSOR FUNCTIONS####
# basic tensor
# x = torch.tensor([2.5, 0.1])


# .zeros and .ones print only single values in matrix


# 1D array
# x = torch.zeros(1)
# print(x)


# 2D array
# x = torch.zeros(1, 2)
# print(x)


# 3D array
# x = torch.ones(1, 2, 3)
# print(x)


# 4D array
# x = torch.ones(1, 2, 3, 4)
# print(x)


#Datatypes
# x = torch.ones(2,2, dtype=torch.float16)
# print(x.type)
# print(x.dtype)
# print(x.size())


# 1D Vector with 3 elements. Number passed into arg represents the number of elements
# x = torch.empty(3)
# print(x)


#Input
# tensor = torch.rand(3,4)
# print(f"Shape of tensor: {tensor.shape}")
# print(f"Datatype of tensor: {tensor.dtype}")
# print(f"Device tensor is stored on: {tensor.device}")
#Output
# Shape of tensor: torch.Size([3, 4])
# Datatype of tensor: torch.float32
# Device tensor is stored on: cpu


# Operations
# x = torch.rand(2, 2)
# y = torch.rand(2, 2)
# Different add operations same result
# z = x + y  
# z = torch.add(x,y)
# Different subtract operations same result
# z = x - y  
# z = torch.sub(x,y)
# Different multiply operations same result
# z = x * y  
# z = torch.mul(x,y)
# Different divisions same result
# z = x / y  
# z = torch.div(x,y)
# print(z)


# inplace operations .add_...
# y.add_(x)
# print(y)


# size manipulation
# x = torch.rand(4,4)
# print(x)
# y = x.view(-1, 8)
# print(y.size())


#####GRADIENTS#####
# x = torch.randn(3, requires_grad=True)
# print(x)
# VISUALIZE THE METHOD
# x        FORWARD->
#    +  y               ] calculated gradients as attribute:
# 2      <-ADD BACKWARD,]  grad_fn
# calculates gradients in y with respect to x in this case
# y = x+2
# print(y)
