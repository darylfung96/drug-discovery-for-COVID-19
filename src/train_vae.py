import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data_processing.dataset import SmilesDataset
from src.models.mmd_vae import MMDVAE
from src.loss import loss_function

batch_size = 64
epoch = 100
verbose_per_step = 1000
device = 'cpu'

if __name__ == '__main__':
    processed_data = np.load('./data_processing/processed_data.npy').astype(np.long)
    with open('./data_processing/tokens.pkl', 'rb') as f:
        data_info = pickle.load(f)

    mmd_vae = MMDVAE(num_inputs=data_info['max_length']).to(device)
    optimizer = Adam(mmd_vae.parameters(), lr=0.001)

    train_smiles_dataset = SmilesDataset(processed_data[:-50000])
    test_smiles_dataset = SmilesDataset(processed_data[50000:])
    train_data_loader = DataLoader(train_smiles_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(test_smiles_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    for current_epoch in range(epoch):
        for idx, train_batch in enumerate(train_data_loader):
            train_batch = train_batch.to(device)
            decoder_output, latent = mmd_vae(train_batch)

            optimizer.zero_grad()
            loss = loss_function(decoder_output, train_batch, latent)
            loss.backward()
            optimizer.step()

            if idx % verbose_per_step == 0:
                print(f'loss: {loss}')

        for idx, test_batch in enumerate(test_data_loader):
            test_batch = test_batch.to(device)
            decoder_output, latent = mmd_vae(test_batch)
            loss = loss_function(decoder_output, test_batch, latent)

            if idx % verbose_per_step == 0:
                print(f'testing loss: {loss}')