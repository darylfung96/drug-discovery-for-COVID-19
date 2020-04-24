import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_kernel(a, b):
    depth = a.shape[1] # hidden size
    a_reshaped = a.view(a.shape[0], 1, depth)
    b_reshaped = b.view(1, b.shape[0], depth)
    a_processed = a_reshaped.expand(a.shape[0], b.shape[0], depth)
    b_processed = b_reshaped.expand(a.shape[0], b.shape[0], depth)
    distance = (a_processed - b_processed).pow(2).mean(2)/depth
    return torch.exp(-distance)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2 * gaussian_kernel(a, b).mean()


def loss_function(prediction, ground_truth, latent, device):
    prediction = prediction.view(-1, 41)

    reconstruction_loss = F.cross_entropy(prediction, ground_truth.view(-1))
    mmd_loss = MMD(torch.randn([200, latent.shape[1]], requires_grad=False).to(device), latent)

    return reconstruction_loss, mmd_loss
