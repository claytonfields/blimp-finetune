# coding=utf-8
import argparse
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import sys

from transformers import ElectraModel
from transformers import ElectraTokenizer
from transformers import AutoTokenizer
from transformers import ElectraForMultipleChoice
from transformers import ElectraTokenizerFast

import torch
from torch import cuda

'''Evaluate an ELECTRA Model on BLiMP with minimal fine-tuning'''


# Torch DataSet Class
class BlimpDataset(torch.utils.data.Dataset):

    def __init__(self, blimp_data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.a = blimp_data['sent_a']
        self.b = blimp_data['sent_b']
        self.labels = blimp_data['label']
        self.len = len(blimp_data)
        self.max_len = max_len

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        a = self.a[index]
        b = self.b[index]
        inputs = self.tokenizer(
            [a, b],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors="pt",
            truncation=False,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }
                
# Torch Utility Functions
def loss_fn(outputs, targets):
    return torch.nn.BCELoss()(outputs, targets)

def train(model, training_loader, optimizer):
    model.train()
    for data in tqdm(training_loader):
      outputs = model(**{k: v.to(device) for k, v in data.items()}, return_dict=True)
      targets = data['labels'].float()
      optimizer.zero_grad()
      loss = loss_fn(torch.sigmoid(outputs['logits'][:,1]), targets.to(device))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    return loss

def validation(model, testing_loader):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for data in tqdm(testing_loader):
            targets = data['labels'].to(device)
            outputs = model(**{k: v.to(device) for k, v in data.items()})
            outputs = torch.sigmoid(outputs['logits']).cpu().detach()
            fin_outputs.extend(outputs)
            fin_targets.extend(targets)
    return torch.stack(fin_outputs), torch.stack(fin_targets)

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
    parser.add_argument("-no_output", '-no',
                        action='store_true',
                        help = "Boolean: write results t output file")
    parser.add_argument('-pretrained','-pt',
                        default = False,
                        type = bool,
                        required = False)
    
    
    # Parse arguments
    args = parser.parse_args()
    model_path = args.model_path
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_length = args.max_length
    no_output = args.no_output
    
    # Check for matching output directory
    if not os.path.exists('output'):
        os.mkdir('output')
    model_name = model_path.rstrip('/')
    model_name = model_name.rstrip('\\')
    model_name = os.path.split(model_name)[1]
    print('model name: ',model_name)
    output_path = os.path.join('output',model_name)
    print('output path: ',output_path)
    if not no_output:
        if os.path.exists(output_path):
            print('output directory already exists')
            sys.exit()

    # os.mkdir(output_path)


    device = 'cuda' if cuda.is_available() else 'cpu'
    # device = 'cpu'
    
    # Data
    if not os.path.exists('data'):
        os.mkdir('data')
    randomized_train_path = os.path.join('data','blimp_train_randomized.csv')
    if os.path.exists(randomized_train_path):
        blimp_train = pd.read_csv(randomized_train_path,index_col=0)
    
    randomized_dev_path = os.path.join('data','blimp_dev_randomized.csv')
    if os.path.exists(randomized_dev_path):
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
    if not no_output:
        os.mkdir(output_path)
    with open(os.path.join(output_path,'eval.txt'),'w') as f:
        for epoch in range(EPOCHS):
            loss = train(model, training_loader, optimizer)
            loss_string = f'Epoch: {epoch}, Loss:  {loss.item()}'
            if not no_output:
                f.write(loss_string)
            print(loss_string)  
            guess, targs = validation(model, dev_loader)
            guesses = torch.max(guess, dim=1)
            targets = torch.max(targs, dim=0)
            acc_string = 'arracy on test set {}'.format(accuracy_score(guesses.indices, targs.cpu()))
            if not no_output:
                f.write(acc_string)
            print(acc_string)
    save_path = os.path.join(output_path, 'pytorch_model.bin')
    if not no_output:
        torch.save(model.state_dict(), save_path)

