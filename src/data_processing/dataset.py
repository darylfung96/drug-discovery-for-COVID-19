from torch.utils.data import Dataset


class SmilesDataset(Dataset):
    def __init__(self, processed_data):
        self.processed_data = processed_data

    def __len__(self):
        return self.processed_data.shape[0]

    def __getitem__(self, idx):
        return self.processed_data[idx]

