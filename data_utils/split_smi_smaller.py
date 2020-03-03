import os

drugs_filename = '../data/drugs_cleansed.smi'
limit_line = 4500

current_count = 0
current_batch = 0

if __name__ == '__main__':
    if not os.path.isdir('../data/drugs_cleansed'):
        os.mkdir('../data/drugs_cleansed')

    if os.path.isfile(drugs_filename):
        with open(drugs_filename, 'r') as f:

            current_content = ''
            for line in f:
                current_content += line
                current_count += 1

                if current_count == limit_line:
                    with open(f'../data/drugs_cleansed/drugs_cleansed_batch_{current_batch * limit_line}-{(current_batch+1) * limit_line}.smi',
                              'w') as drugs_file:
                        drugs_file.write(current_content)
                        current_content = ''
                        current_count = 0
                        current_batch += 1
