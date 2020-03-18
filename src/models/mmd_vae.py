import torch
import torch.nn as nn


class MMDVAE(nn.Module):
    def __init__(self, num_inputs):
        super(MMDVAE, self).__init__()
        self.embedding = nn.Embedding(num_inputs, 128)


    def forward(self, inputs):
        torch_tensor = torch.from_numpy(inputs)
        self.embedding(torch_tensor)