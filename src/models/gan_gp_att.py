import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms

from torch.autograd import Variable
import torch.optim as optim

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

"""

"""

device = 'cpu:0'


class Generator(nn.Module):
    def __init__(self, z_dim, vocab_size):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size

        self.pre_model = nn.Sequential(nn.Linear(z_dim, 64),
                                       nn.LeakyReLU(0.2, inplace=True),
                                       nn.Linear(64, 32 * 8 * 8)
                                       )

        self.model = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 5, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, 5, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 5, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 22, 5, stride=1, padding=1),
            nn.BatchNorm2d(22, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.gru = nn.GRU(3200+vocab_size, 128, 1, batch_first=True)
        # self.post_gru = nn.Linear(128, vocab_size)

    def forward(self, x):
        z = self.pre_model(x)
        post_z = z.view(x.shape[0], 32, 8, 8)
        conv_z = self.model(post_z)
        return conv_z.view(x.shape[0], -1)
        # latent_input = conv_z.view(x.shape[0], -1)
        # gru_input = torch.zeros((x.shape[0], self.vocab_size)).to(device)
        # hidden = self.initialize_hidden_state(x)

        # output_sequence = gru_input.unsqueeze(1)
        # for _ in range(80):
        #     gru_input = torch.cat((latent_input, gru_input), dim=1)
        #     gru_input = gru_input.unsqueeze(1)
        #     gru_output, hidden = self.gru(gru_input, hidden)
        #     gru_output = self.post_gru(gru_output.squeeze(1))
        #     gru_input = gru_output
        #     output_sequence = torch.cat((output_sequence, gru_input.unsqueeze(1)), dim=1)

        # return output_sequence

    def initialize_hidden_state(self, x):
        batch_size = x.shape[0]
        hidden_state = torch.zeros((1, batch_size, 128))
        return hidden_state


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        def block(inputs, outputs, bn=True):
            # padding = kernel // 2
            c_block = [nn.Conv2d(inputs, outputs, 3, 2, padding=3//2), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                c_block.append(nn.BatchNorm2d(outputs, 0.8))
            return c_block

        self.model = nn.Sequential(
            *block(22, 16, bn=False),
            *block(16, 32, bn=False),
            *block(32, 64, bn=False),
            *block(64, 64, bn=False),
            *block(64, 1, bn=False)
        )

        # self.last_layer = nn.Sequential(nn.Linear(192, 1))

    def forward(self, x):
        x = x.view(x.shape[0], 22, 4, 4)
        output = self.model(x)
        output = output.view(x.shape[0], -1)
        # output = self.last_layer(output)
        return output





# generator = Generator(Z_DIM, 256, X_SIZE).to(device)
# discriminator = Discriminator(X_SIZE, 256).to(device)
# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)
#
#
# d_optim = optim.Adam(discriminator.parameters(), lr=0.0002, betas=[0.5, 0.999])
# g_optim = optim.Adam(generator.parameters(), lr=0.0002, betas=[0.5, 0.999])
#
#
# picture_out_count = 0
#
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "./data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True
# )
#
# for epoch in range(200):
#     for i, (imgs, _) in enumerate(dataloader):
#         imgs = imgs.to(device)
#
#         d_optim.zero_grad()
#
#         # create fake inputs
#         g_ = generator(Variable(torch.randn(BATCH_SIZE, Z_DIM)).to(device))
#
#         #pass input through discriminator
#         d_real_value = discriminator(imgs)
#         d_real_error = -torch.mean(d_real_value)
#
#         #pass fake through discriminator
#         d_fake_value = discriminator(g_.detach())
#         d_fake_error = torch.mean(d_fake_value)
#
#         gp = gradient_penalty(discriminator, imgs, g_)
#
#         d_total_error = d_real_error + d_fake_error + 10 * gp
#         d_total_error.backward()
#         d_optim.step()
#
#
#         if i % 5 == 0:
#             # train generator
#             g_optim.zero_grad()
#             g_ = generator(Variable(torch.randn(BATCH_SIZE, Z_DIM)).to(device))
#             g_value = discriminator(g_)
#             g_error = -torch.mean(g_value)
#             g_error.backward()
#             g_optim.step()
#
#             samples = g_[:16].data.cpu().numpy() # get 16 pictures
#             fig = plt.figure(figsize=(4, 4))
#             gs = gridspec.GridSpec(4, 4)
#             gs.update(wspace=0.05, hspace=0.05)
#
#             for i, sample in enumerate(samples):
#                 ax = plt.subplot(gs[i])
#                 plt.axis('off')
#                 ax.set_xticklabels([])
#                 ax.set_yticklabels([])
#                 ax.set_aspect('equal')
#                 plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#
#             if not os.path.exists('out/'):
#                 os.makedirs('out/')
#
#             plt.savefig('out/{}.png'.format(str(picture_out_count).zfill(3)))
#             picture_out_count+=1
#             plt.close(fig)
#
#             print(f'epoch: {epoch} d_loss: {d_total_error} g_loss: {g_error}')
#
#
#
#
#
#
