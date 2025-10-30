import torch
import torch.nn.functional as F

all_targets = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1, 0 , 0 , 0 , 0], [1, 0, 0, 1, 0, 0, 0, 0, 0 , 0 , 0 , 1]]).float()
all_targets = F.normalize(all_targets, p=1, dim=1)

print(all_targets)
