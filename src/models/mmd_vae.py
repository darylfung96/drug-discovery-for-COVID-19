import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import os

from src.mmd_loss import loss_function


class MMDVAE(nn.Module):
    def __init__(self, num_inputs, load_path):
        super(MMDVAE, self).__init__()
        self.num_inputs = num_inputs
        self.embedding = nn.Embedding(num_inputs, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 128)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=(7, 1))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 80, (4, 1))
        )

        if os.path.isfile(load_path):
            with open(load_path, 'r') as f:
                self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    def forward(self, inputs):
        output = self.embedding(inputs)
        output = output.unsqueeze(1)
        encoder_output = self.encoder(output)
        final_output, latent = self.forward_decoder(encoder_output)

        return final_output, latent

    def forward_decoder(self, encoder_output):
        latent = encoder_output.view(encoder_output.shape[0], -1)
        final_output = self.decoder(encoder_output)
        # decoder_output = F.softmax(decoder_output, dim=2)
        # decoder_output = decoder_output.view(encoder_output.shape[0], 1, -1)
        # decoder_output = decoder_output.repeat(1, 80, 1)  # batch_size, max_length, num_chars
        # out, hidden = self.gru(decoder_output)
        # final_output = self.final_layer(out)
        return final_output, latent

    def sample(self, latent=None):
        latent = latent if latent is not None else torch.randn(1, 256).to(self.device)
        latent = latent.view(-1, 8, 32, 1)
        decoder_output = F.softmax(self.forward_decoder(latent)[0].view(-1, 41), 1)
        return decoder_output.view(-1, 80, 41)
