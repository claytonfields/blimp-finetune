# coding=utf-8
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import sys

from transformers import ElectraForMultipleChoice
from transformers import ElectraTokenizer
from transformers import ElectraTokenizerFast

import torch
from torch import cuda

from model_utils import BlimpDataset
from model_utils import loss_fn, train, validation
from data_processing import process_data



'''Evaluate an ELECTRA Model on BLiMP with a supervised finetuning regime.'''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ELECTRA Info')
    parser.add_argument('model_path',
                        default = None,
                        type = str,
                        help = "Path to the PyTorch checkpoint.")
    parser.add_argument("-epochs", '-e',
                        default = 1,
                        type = int,
                        required = False,
                        help = "Number of training epochs.")
    parser.add_argument("-learning_rate", '-lr',
                        default = 2e-05,
                        type = float,
                        required = False,
                        help = 'Optimizer learning rate.')
    parser.add_argument("-max_length", '-ml',
                        default = 128,
                        type = int,
                        required = False,
                        help = "Maximum sequence length.")
    parser.add_argument("-batch_size", '-bs',
                        default = 32,
                        type = int,
                        required = False,
                        help = "Training and eval batch size.")
    parser.add_argument("-prop_train", '-pt',
                        default = .10,
                        type = float,
                        required = False,
                        help = 'Proportion of data to use for training, between 0 and 1')
    
    
    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_length = args.max_length
    prop_train = args.prop_train

    
    # Check for matching output directory
    if not os.path.exists('output'):
        os.mkdir('output')
    model_name = model_path.rstrip('/')
    model_name = model_name.rstrip('\\')
    model_name = os.path.split(model_name)[1]
    print('model name: ',model_name)
    output_path = os.path.join('output',model_name)
    print('output path: ',output_path)
    if os.path.exists(output_path):
        print('output directory already exists')
        sys.exit()

    # Default data split
    if prop_train == 0.10:
        randomized_train_path = os.path.join('data', 'default','blimp_train_randomized.csv')
        randomized_dev_path = os.path.join('data','default','blimp_dev_randomized.csv')
    # Custom datasplit
    else:
        randomized_train_path, randomized_dev_path = process_data(prop_train)
    blimp_train = pd.read_csv(randomized_train_path,index_col=0)
    blimp_dev = pd.read_csv(randomized_dev_path, index_col=0)
    
    # Choose device
    device = 'cuda' if cuda.is_available() else 'cpu'

    # Model and Tokenizer
    model = ElectraForMultipleChoice.from_pretrained(model_path)
    
    if os.path.exists(os.path.join(model_path,'tokenizer.json')):
        print('Using local tokenizer.json file.')
        tokenizer_name = os.path.join(model_path,'tokenizer.json')
        tokenizer = ElectraTokenizerFast(tokenizer_file=tokenizer_name)
    else:    
        tokenizer_name = 'google/electra-small-discriminator'
        tokenizer = ElectraTokenizer.from_pretrained(tokenizer_name)

    # Train Params
    MAX_LEN = max_length
    BATCH_SIZE = batch_size
    EPOCHS = epochs
    LEARNING_RATE = learning_rate

    # Datasets and Dataloaders
    training_data = BlimpDataset(blimp_train, tokenizer, MAX_LEN)
    dev_data = BlimpDataset(blimp_dev, tokenizer, MAX_LEN)

    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    dev_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }    

    training_loader = torch.utils.data.DataLoader(training_data, **train_params)
    dev_loader = torch.utils.data.DataLoader(dev_data, **dev_params)
    
    # Training and Eval loop
    model.to(device)    
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)
    os.mkdir(output_path)
    with open(os.path.join(output_path,'eval.txt'),'w') as f:
        for epoch in range(EPOCHS):
            loss = train(model, training_loader, optimizer, device)
            loss_string = f'Epoch: {epoch}, Loss:  {loss.item()} \n'
            f.write(loss_string)
            print(loss_string)  
            guess, targs = validation(model, dev_loader, device)
            guesses = torch.max(guess, dim=1)
            targets = torch.max(targs, dim=0)
            acc_string = 'acurracy on test set {}'.format(accuracy_score(guesses.indices, targs.cpu()))
            f.write(acc_string)
            print(acc_string)
    save_path = os.path.join(output_path, 'pytorch_model.bin')

    torch.save(model.state_dict(), save_path)

