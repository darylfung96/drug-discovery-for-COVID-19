import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os


class VAE(nn.Module):
    def __init__(self, num_inputs, max_length, device, load_path):
        super(VAE, self).__init__()
        self.num_inputs = num_inputs
        self.max_length = max_length
        self.device = device
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

        self.latent_mean = nn.Linear(256, 256)
        self.latent_std = nn.Linear(256, 256)

        self.pre_decoder = nn.Linear(256, 256)
        self.gru_decoder = nn.GRU(256, 1024, 3, batch_first=True)
        self.post_decoder = nn.Linear(1024, self.num_inputs)

        self.mean = None
        self.std = None

        self.criterion = nn.NLLLoss()

        if os.path.isfile(load_path):
            with open(load_path, 'r') as f:
                self.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

    def forward(self, inputs):
        output = self.embedding(inputs)
        output = output.unsqueeze(1)
        encoder_output = self.encoder(output)
        reshaped_encoder_output = encoder_output.view(inputs.shape[0], -1)
        self.mean = self.latent_mean(reshaped_encoder_output)
        self.std = self.latent_std(reshaped_encoder_output)

        eps = 1e-2 * Variable(torch.randn(*self.std.size())
                                    ).type_as(self.std)
        latent = self.mean + torch.exp(self.std / 2.) * eps

        final_output = self.forward_decoder(latent)
        # final_output, latent = self.forward_decoder(pre_decoder)

        return final_output, latent

    def forward_decoder(self, latent):
        pre_decoder = self.pre_decoder(latent)
        gru_input = pre_decoder.view(latent.shape[0], -1).unsqueeze(1).repeat(1, self.max_length, 1)
        gru_output, h = self.gru_decoder(gru_input)
        final_output = self.post_decoder(gru_output)
        # decoder_output = F.softmax(decoder_output, dim=2)
        # decoder_output = decoder_output.view(encoder_output.shape[0], 1, -1)
        # decoder_output = decoder_output.repeat(1, 80, 1)  # batch_size, max_length, num_chars
        # out, hidden = self.gru(decoder_output)
        # final_output = self.final_layer(out)
        return final_output

    def calculate_loss(self, prediction, ground_truth, latent):
        ground_truth_one_hot = torch.nn.functional.one_hot(ground_truth, self.num_inputs).detach().type(torch.FloatTensor)
        prediction = F.log_softmax(prediction, dim=2)
        # reconstruction_loss = self.criterion(prediction, ground_truth_one_hot)
        reconstruction_loss = -torch.sum(ground_truth_one_hot * prediction)
        kl_loss = -0.5 * torch.mean(1. + self.std - self.mean ** 2. - torch.exp(self.std))
        return reconstruction_loss, kl_loss

    def sample(self, latent=None):
        latent = latent if latent is not None else torch.randn(1, 256).to(self.device)
        decoder_output = F.softmax(self.forward_decoder(latent)[0].view(-1, self.num_inputs), 1)
        return decoder_output.view(-1, self.max_length, self.num_inputs)
