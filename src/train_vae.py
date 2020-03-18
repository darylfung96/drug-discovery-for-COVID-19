import numpy as np
import pickle

if __name__ == '__main__':
    processed_data = np.load('./data_processing/processed_data.npy')
    with open('./data_processing/tokens.pkl', 'rb') as f:
        data_info = pickle.load(f)
    print('a')