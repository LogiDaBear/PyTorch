# Starting a PyTorch instance - tutorial


#### Author: Logan Reese


### What the cuss is a tensor and why do I care?
Glad  you asked! A tensor is a multi-dimensional array that can be used to represent data in a variety of ways. In software development, tensors are often used to represent images, videos, and other types of data that have multiple dimensions.


For example, an image can be represented as a 3D tensor, with the first dimension representing the width of the image, the second dimension representing the height of the image, and the third dimension representing the color channels (red, green, and blue).


Tensors can also be used to represent more complex data structures, such as neural networks. Neural networks are a type of machine learning algorithm that can be used to solve a variety of problems, including image classification, natural language processing, and speech recognition.


Tensors are a powerful tool for representing and manipulating data in software development. They are used in a variety of different applications, including machine learning, computer vision, and natural language processing.


***If you stored the tensor on the CPU not the GPU be careful - both objects will share the same memory location, this means if you change one you will change the other. See below example***
```
import torch
import numpy as np


a = torch.ones(5)
b = a.numpy()
print(b)


a.add_1(1)
print(a)
print(b)
```
Helpful links:
- [Tensors in PyTorch](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)
- [Tensors for Beginners](https://www.tensorflow.org/guide/tensor)
### This is a step by step tutorial project so you can hit the ground running with PyTorch **Note: all the terminal commands I enter are using Ubuntu**
 


Step 1: Create a virtual environment
- python3 -m venv .venv


**ATTENTION:**
When installing pay close attention to the versions- you may require a legacy version of CUDA for PyTorch


Step 2: Install [NvidiaCUDA](https://developer.nvidia.com/cuda-downloads) toolkit and [PyTorch](https://pytorch.org/get-started/locally/)


Step 3: Verify installation of PyTorch and CUDA toolkits with these terminal commands:
- `python`, `import torch`, for testing enter: `x = torch.rand(3)`, `print(x)`- here you should get a tensor result back. Check if cuda is available with `torch.cuda.is_available()`- this will return True or False depending on whether you installed CPU cuda or not.
- For CPU install: pip3 install torch torchvision torchaudio


Step 4: Familiarize the [Cheat Sheet](https://pytorch.org/tutorials/beginner/ptcheat.html) for specific functionality




### Contribution
- [Deep Learning w/ PyTorch](https://www.youtube.com/watch?v=c36lUUr864M&t=526s)

