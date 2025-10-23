import torch
import torch.nn as nn

x = torch.tensor([[[[1,2,3,4,5],
                    [6,7,8,9,10],
                    [11,12,13,14,15],
                    [16,17,18,19,20],
                    [21,22,23,24,25]]]], dtype=torch.float32)

pool = nn.AdaptiveAvgPool2d((2, 2))
output = pool(x)
print(output)
