import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os

from src.trainer.trainer import Trainer
from src.models.mmd_vae import MMDVAE
from src.models.vae import VAE


model_type_dict = {
    'mmd': MMDVAE,
    'vae': VAE
}


class VAETrainer(Trainer):
    def __init__(self, data_info, model_type, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device):
        super(VAETrainer, self).__init__(data_info, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device)
        self.vae = model_type_dict[model_type](num_inputs=len(data_info['tokens_dict']), max_length=data_info['max_length'],
                                                   device=device, load_path='./models/model.ckpt').to(device)
        self.writer = SummaryWriter('runs/experiment')
        # self.writer.add_graph(self.vae, list(self.train_data_loader)[0].to(self.device))

    def run(self):
        optimizer = Adam(self.vae.parameters(), lr=0.0005)
        running_loss = 0
        running_index = 0
        best_reconstruction_loss = float('inf')
        self.vae.train()
        current_model_name = ''

        for current_epoch in range(self.epoch):
            for idx, train_batch in enumerate(self.train_data_loader):
                train_batch = train_batch.to(self.device)
                decoder_output, latent = self.vae(train_batch)

                optimizer.zero_grad()
                reconstruction_loss, latent_loss = self.vae.calculate_loss(decoder_output, train_batch, latent)
                weighted_latent_loss = latent_loss
                loss = reconstruction_loss + weighted_latent_loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_index += 1

                if running_index % 1000 == 0:
                    self.writer.add_scalar('training_loss', running_loss / 1000, running_index)
                    running_loss = 0

                if idx % self.verbose_per_step == 0 and reconstruction_loss < best_reconstruction_loss:
                    print(
                        f'epoch: {current_epoch} reconstruction loss: {reconstruction_loss} '
                        f'latent loss: {weighted_latent_loss}')

                if idx % self.save_per_step == 0 and reconstruction_loss:

                    if os.path.isfile(current_model_name):
                        os.remove(current_model_name)
                    current_model_name = f'./models/model_{current_epoch}.ckpt'
                    torch.save(self.vae.state_dict(), current_model_name)

            # for idx, test_batch in enumerate(test_data_loader):
            #     test_batch = test_batch.to(device)
            #     decoder_output, latent = mmd_vae(test_batch)
            #     reconstruction_loss, mmd_loss = loss_function(decoder_output, test_batch, latent)
            #
            #     if idx % verbose_per_step == 0:
            #         print(f'testing recons loss: {reconstruction_loss} testing mmd loss: {mmd_loss}')

        self.vae.eval()
        #  sample generation
        print("generating with random latents")
        output = self.vae.sample()
        generated_smiles_indexes = torch.argmax(output, 2).cpu().detach().numpy()
        generated_smiles = []
        for i in range(len(generated_smiles_indexes)):
            value = [self.data_info['index_dict'][idx] for idx in generated_smiles_indexes[i]]
            generated_smiles.append(''.join(value).replace('0', ''))
        print(generated_smiles)

        # sample existing latent generation
        print('generating with existing latents')
        output = self.vae.sample(latent)
        generated_smiles_indexes = torch.argmax(output, 2).cpu().detach().numpy()
        generated_smiles = []
        for i in range(len(generated_smiles_indexes)):
            value = [self.data_info['index_dict'][idx] for idx in generated_smiles_indexes[i]]
            generated_smiles.append(''.join(value).replace('0', ''))
        print(generated_smiles)

        # show real data
        print('real data')
        numpy_train_batch = train_batch.cpu().detach().numpy()
        generated_smiles = []
        for i in range(numpy_train_batch.shape[0]):
            value = [self.data_info['index_dict'][idx] for idx in numpy_train_batch[i]]
            generated_smiles.append(''.join(value))
        print(generated_smiles)

