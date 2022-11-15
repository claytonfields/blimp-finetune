import torch
from tqdm import tqdm

'''
Utility functions for training a pytorch model and a custom pyorch DataSet.
'''


#  Torch DataSet Class
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

def train(model, training_loader, optimizer, device):
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

def validation(model, testing_loader, device):
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