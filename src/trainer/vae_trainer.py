import torch
from torch.optim import Adam

from src.trainer.trainer import Trainer
from src.mmd_loss import loss_function
from src.models.mmd_vae import MMDVAE


model_type_dict = {
    'mmd': MMDVAE
}


class VAETrainer(Trainer):
    def __init__(self, data_info, model_type, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device):
        super(VAETrainer, self).__init__(data_info, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device)
        self.mmd_vae = model_type_dict[model_type](num_inputs=data_info['max_length'], load_path='./models/model.ckpt').to(device)

    def run(self):
        optimizer = Adam(self.mmd_vae.parameters(), lr=0.0005)

        for current_epoch in range(self.epoch):
            for idx, train_batch in enumerate(self.train_data_loader):
                train_batch = train_batch.to(self.device)
                decoder_output, latent = self.mmd_vae(train_batch)

                optimizer.zero_grad()
                reconstruction_loss, mmd_loss = loss_function(decoder_output, train_batch, latent)
                weighted_mmd_loss = mmd_loss
                loss = reconstruction_loss + weighted_mmd_loss
                loss.backward()
                optimizer.step()

                if idx % self.verbose_per_step == 0:
                    print(
                        f'epoch: {current_epoch} reconstruction loss: {reconstruction_loss} mmd loss: {weighted_mmd_loss}')

                if idx % self.save_per_step == 0:
                    torch.save(self.mmd_vae.state_dict(), './models/model.ckpt')

            # for idx, test_batch in enumerate(test_data_loader):
            #     test_batch = test_batch.to(device)
            #     decoder_output, latent = mmd_vae(test_batch)
            #     reconstruction_loss, mmd_loss = loss_function(decoder_output, test_batch, latent)
            #
            #     if idx % verbose_per_step == 0:
            #         print(f'testing recons loss: {reconstruction_loss} testing mmd loss: {mmd_loss}')

        #  sample generation
        print("generating with random latents")
        output = self.mmd_vae.sample()
        generated_smiles_indexes = torch.argmax(output, 2).numpy()
        generated_smiles = []
        for i in range(len(generated_smiles_indexes)):
            value = [self.data_info['index_dict'][idx] for idx in generated_smiles_indexes[i]]
            generated_smiles.append(''.join(value).replace('0', ''))
        print(generated_smiles)

        # sample existing latent generation
        print('generating with existing latents')
        output = self.mmd_vae.sample(latent)
        generated_smiles_indexes = torch.argmax(output, 2).numpy()
        generated_smiles = []
        for i in range(len(generated_smiles_indexes)):
            value = [self.data_info['index_dict'][idx] for idx in generated_smiles_indexes[i]]
            generated_smiles.append(''.join(value).replace('0', ''))
        print(generated_smiles)

        # show real data
        print('real data')
        numpy_train_batch = train_batch.numpy()
        generated_smiles = []
        for i in range(numpy_train_batch.shape[0]):
            value = [self.data_info['index_dict'][idx] for idx in numpy_train_batch[i]]
            generated_smiles.append(''.join(value))
        print(generated_smiles)

