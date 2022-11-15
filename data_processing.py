import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import math
import os

'''
Function for randomizing a train/test split of the BLiMP dataset.
'''

def process_data(frac_train):

    n_train = math.floor(1000*frac_train)
    data_folder_path = os.path.join('data', str(frac_train))

    if not os.path.exists(data_folder_path):
        os.mkdir(data_folder_path)
    randomized_train_path = os.path.join(data_folder_path,'blimp_train_randomized.csv')
    randomized_dev_path = os.path.join(data_folder_path,'blimp_dev_randomized.csv')

    if not (os.path.exists(randomized_dev_path) and os.path.exists(randomized_train_path)):
        file_paths = glob.glob("data/BLiMP/*.jsonl")

        print(f'Processing data files for {frac_train}/{1-frac_train} train/test split')
        blimp_train_data = pd.DataFrame()
        blimp_dev_data = pd.DataFrame()
        for path in tqdm(file_paths):
            temp_data = pd.read_json(path, lines=True)
            temp_train = temp_data[:n_train]
            temp_dev = temp_data[n_train:]
            blimp_train_data = blimp_train_data.append(temp_train,ignore_index=True)
            blimp_dev_data = blimp_dev_data.append(temp_dev,ignore_index=True)

        print(f'Creating randomized train data split for {frac_train}/{1-frac_train} train/test split')
        blimp_train = pd.DataFrame({'sent_a':[],'sent_b':[],'label':[]})
        for idx in tqdm(blimp_train_data.index):
            if np.random.uniform() < 0.5:
                blimp_train.loc[idx] = [blimp_train_data.at[idx,'sentence_good'],blimp_train_data.at[idx,'sentence_bad'],0]
            else:
                blimp_train.loc[idx]=[blimp_train_data.at[idx,'sentence_bad'],blimp_train_data.at[idx,'sentence_good'],1]
        blimp_train.to_csv(randomized_train_path)

        print(f'Creating randomized test data split for {frac_train}/{1-frac_train} train/test split')
        blimp_dev = pd.DataFrame({'sent_a':[],'sent_b':[],'label':[]})
        for idx in tqdm(blimp_dev_data.index):
            if np.random.uniform() < 0.5:
                blimp_dev.loc[idx] = [blimp_dev_data.at[idx,'sentence_good'],blimp_dev_data.at[idx,'sentence_bad'],0]
            else:
                blimp_dev.loc[idx]=[blimp_dev_data.at[idx,'sentence_bad'],blimp_dev_data.at[idx,'sentence_good'],1]
        blimp_dev.to_csv(randomized_dev_path)

    return randomized_train_path, randomized_dev_path