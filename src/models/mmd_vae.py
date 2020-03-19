import torch
import torch.nn as nn


class MMDVAE(nn.Module):
    def __init__(self, num_inputs):
        super(MMDVAE, self).__init__()
        self.num_inputs = num_inputs
        self.embedding = nn.Embedding(num_inputs, 128)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 128)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=(7, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=(5, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 1))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, (3, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, (5, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 80, (5, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(80, 80, (5, 1)),
            nn.ReLU(),
            nn.Conv2d(80, 80, (7, 1)),
            nn.ReLU(),
            nn.Conv2d(80, 80, (7, 1)),
            nn.ReLU(),
            nn.Conv2d(80, 80, (5, 1)),
            nn.ReLU(),
            nn.Conv2d(80, 80, (4, 1)),
            nn.Sigmoid()

        )

    def forward(self, inputs):
        output = self.embedding(inputs)
        output = output.unsqueeze(1)
        encoder_output = self.encoder(output)
        latent = encoder_output.view(inputs.shape[0], -1)
        decoder_output = self.decoder(encoder_output)
        return decoder_output, latent



