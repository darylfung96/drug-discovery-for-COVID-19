import numpy as np
import pickle


class SmilesProcessing:
    def __init__(self, smiles_filename, token_save_filename, data_save_filename):
        self.smiles_filename = smiles_filename
        self.token_save_filename = token_save_filename
        self.data_save_filename = data_save_filename
        with open(self.smiles_filename, 'r') as f:
            self.content = f.read()
            self.line_content = self.content.split("\n")
            self.max_length = len(max(self.line_content, key=len))
        self.tokens = None
        self.tokens_dict = None
        self.padded_content = None
        self.formatted_content = []

    def get_unique_tokens(self):
        self.tokens = ['0'] + list(set(self.content.replace('\n', '')))  # '0' for end of string
        self.tokens = np.array(self.tokens)
        self.tokens_dict = {token: index for index, token in enumerate(self.tokens)}
        self.index_dict = {index: token for index, token in enumerate(self.tokens)}

    def convert_to_index(self):
        for line_content in self.line_content:
            self.formatted_content.append([self.tokens_dict[letter] for letter in list(line_content)])

    def pad_content(self):
        self.padded_content = np.array([np.pad(formatted_content, (0, self.max_length - len(formatted_content)), 'constant')
                               for formatted_content in self.formatted_content])
        self.padded_content = self.padded_content.astype(np.int32)

    def save(self):
        with open(self.token_save_filename, 'wb') as f:
            pickle.dump({'max_length': self.max_length, 'tokens_dict': self.tokens_dict, 'index_dict': self.index_dict},
                        f)
        np.save(self.data_save_filename, self.padded_content)

    def process(self):
        self.get_unique_tokens()
        self.convert_to_index()
        self.pad_content()
        self.save()


if __name__ == '__main__':
    smiles_processing = SmilesProcessing('../../data/drugs_cleansed.smi', './tokens.pkl',
                                         './processed_data.npy')
    smiles_processing.get_unique_tokens()
    smiles_processing.convert_to_index()
    smiles_processing.pad_content()
    smiles_processing.save()
