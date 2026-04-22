import torch
import torch.nn.functional as F

all_targets = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 1, 0 , 0 , 0 , 0], [1, 0, 0, 1, 0, 0, 0, 0, 0 , 0 , 0 , 1]]).float()
a = F.normalize(all_targets, p=1, dim=1)

b = F.log_softmax(all_targets, dim=1)
print(a)
print(b)
