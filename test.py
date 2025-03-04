import torch
import torchsearchsorted

# Example 1-D tensors
cdf = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], device='cuda')
u = torch.tensor([0.15, 0.35], device='cuda')

# Reshape tensors to be 2-D
cdf = cdf.unsqueeze(0)  # Convert to shape (1, 5)
u = u.unsqueeze(0)      # Convert to shape (1, 2)

# Perform searchsorted operation
inds = torchsearchsorted.searchsorted(cdf, u, side="right")
print(inds)
