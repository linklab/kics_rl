import torch
import torch.nn as nn

import torch.nn as nn

batch_size = 3

feature_dim = 256
out_dim = 8

time_step = 16000

x = torch.rand(batch_size, feature_dim, time_step)
print('input_size:', x.shape)

conv1d = nn.Conv1d(in_channels=feature_dim, out_channels=out_dim, kernel_size=1)
print('kernel_size:', conv1d.weight.shape)

out = conv1d(x)
print('output_size:', out.shape)