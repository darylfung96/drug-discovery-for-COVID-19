import torch
from torch.autograd import Variable
from torch.optim import Adam

from src.trainer.trainer import Trainer
from src.models.gan_gp_att import Generator, Discriminator

latent_size = 100
device = "cpu:0"

model_type_dict = {
    'dcgan_gp': [Generator, Discriminator]
}


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def gradient_penalty(discrim, real_inputs, fake_inputs):
    alpha = torch.randn((real_inputs.size(0), 1, 1)).to(device)
    interpolates = ((alpha) * real_inputs + (1 - alpha) * fake_inputs).requires_grad_(True)
    d_interpolates = discrim(interpolates)
    target = Variable(torch.ones((real_inputs.size(0), 1)), requires_grad=False).to(device)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=target, create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


class GANTrainer(Trainer):
    def __init__(self, data_info, model_type, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device):
        super(GANTrainer, self).__init__(data_info, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device)

        self.max_length = data_info['max_length']  # max length for chemical molecular smiles
        self.vocab_size = len(data_info['tokens_dict'])

        generator_constructor, discriminator_constructor = model_type_dict[model_type]
        self.generator = generator_constructor(latent_size, self.vocab_size)
        self.discriminator = discriminator_constructor(self.max_length, self.vocab_size)

        self.generator_optim = Adam(self.generator.parameters(), lr=0.0001, betas=[0.5, 0.999])
        self.discriminator_optim = Adam(self.discriminator.parameters(), lr=0.0004, betas=[0.5, 0.999])

    def run(self):
        for current_epoch in range(self.epoch):
            for idx, train_batch in enumerate(self.train_data_loader):
                train_batch = torch.nn.functional.one_hot(train_batch, self.vocab_size).float()
                latent = Variable(torch.randn((train_batch.shape[0], latent_size))).to(device)

                # discriminator
                self.discriminator_optim.zero_grad()
                g_ = self.generator(latent)  # [:, 1:, :]
                d_fake = self.discriminator(g_)
                d_real = self.discriminator(train_batch)
                gp = gradient_penalty(self.discriminator, train_batch, g_.detach())
                discrim_loss = -torch.mean(d_real) + torch.mean(d_fake) + 10 * gp
                discrim_loss.backward()
                self.discriminator_optim.step()

                if idx % 5 == 0:
                    # train generator
                    self.generator_optim.zero_grad()
                    g_ = self.generator(latent)  # [:, 1:, :]
                    d_fake = self.discriminator(g_)
                    generator_loss = -torch.mean(d_fake)
                    generator_loss.backward()
                    self.generator_optim.step()

                print(f"epoch: {current_epoch}, batch:{idx}|{len(train_batch)} generator_loss: {generator_loss}, "
                      f"discriminator_loss: {discrim_loss}")
                print(''.join([self.data_info['index_dict'][value] for value in g_.detach().numpy().argmax(2)[0]]))
                print(''.join([self.data_info['index_dict'][value] for value in train_batch.numpy().argmax(2)[0]]))
        print(g_)