from abc import ABC, abstractmethod


class Trainer(ABC):
    def __init__(self, data_info, epoch, train_data_loader, test_data_loader, verbose_per_step, save_per_step, device):
        self.data_info = data_info
        self.epoch = epoch
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.verbose_per_step = verbose_per_step
        self.save_per_step = save_per_step
        self.device = device

    @abstractmethod
    def run(self):
        pass


