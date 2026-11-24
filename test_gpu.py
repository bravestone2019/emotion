import torch

device = torch.device("cuda")

a = torch.randn(3000, 3000, device=device)
b = torch.randn(3000, 3000, device=device)

c = torch.matmul(a, b)

print("Device:", c.device)
print("Calculation successful!")
print("Result shape:", c.shape)