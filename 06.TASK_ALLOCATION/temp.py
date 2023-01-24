import torch

# Create an example tensor A
A = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Create an example tensor B
B = torch.tensor([[True, False, True], [True, True, False], [True, True, True]])

if B[2, :].all():
    print(B[2, :])

# Check if the second dimension values of tensor B are all true
if not B[:, 0:].all():
    # Fill the tensor A with -1 if any value in the second dimension of tensor B is not true
    A[:, 1:2] = float('inf')

print(A)