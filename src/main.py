import numpy as np
import pickle
from torch.utils.data import DataLoader

from src.data_processing.dataset import SmilesDataset
from src.trainer.vae_trainer import VAETrainer
from src.trainer.gan_trainer import GANTrainer

batch_size = 2
epoch = 200
verbose_per_step = 2
save_per_step = 100
device = 'cpu'
type = 'mmd'

trainer_dict = {
    "mmd": VAETrainer,
    "dcgan_gp": GANTrainer
}

if __name__ == '__main__':
    processed_data = np.load('./data_processing/processed_data.npy').astype(np.long)
    with open('./data_processing/tokens.pkl', 'rb') as f:
        data_info = pickle.load(f)
    train_smiles_dataset = SmilesDataset(processed_data[:2])
    test_smiles_dataset = SmilesDataset(processed_data[2:256])
    train_data_loader = DataLoader(train_smiles_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_data_loader = DataLoader(test_smiles_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    trainer = trainer_dict[type](data_info, type, epoch, train_data_loader, test_data_loader,
                                 verbose_per_step, save_per_step, device)
    trainer.run()
