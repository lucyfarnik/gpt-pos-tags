#%%
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from gather_heads_pos import model, childrens_tagged_sents, reconstruct_sentence, get_pos_dict
from explore_attn_results import get_pos_positions

MAIN = __name__ == '__main__'
#%%
# WHICH LAYER TO USE
activation_name = 'resid_post'
layer = 1
# %%
# get the activation point caches and pos tags for each sentence
if MAIN:
    filename = f'{activation_name}{layer}.pt'
    if os.path.exists(filename):
        data = t.load(filename)
    else:
        data = []
        for sent_idx, tagged_sent in enumerate(childrens_tagged_sents):
            print(sent_idx)
            # Ignore short sentences
            if len(tagged_sent) < 15:
                continue

            sent = reconstruct_sentence(tagged_sent)
            pos_dict = get_pos_dict(sent, tagged_sent)

            _, cache = model.run_with_cache(sent) #TODO make this run on CUDA/MPS

            data.append({
                'sent': sent,
                'pos_dict': pos_dict,
                'act': cache[activation_name, layer],
            })

        t.save(data, filename)
#%%
# gather the activations for different POS tags together
if MAIN:
    noun_activations = []
    verb_activations = []
    adj_activations = []
    other_activations = []
    for sent in data:
        for i, pos in sent['pos_dict'].items():
            # TODO consider adding prepositions since L2H8 also seems to like those
            if pos.startswith('NN'):
                noun_activations.append(sent['act'][0, i])
            elif pos.startswith('VB'):
                verb_activations.append(sent['act'][0, i])
            elif pos.startswith('JJ'):
                adj_activations.append(sent['act'][0, i])
            else:
                other_activations.append(sent['act'][0, i])
#%%
# create train and test sets to train the linear classifier
if MAIN:
    X = t.cat([t.stack(noun_activations), t.stack(verb_activations),
               t.stack(adj_activations), t.stack(other_activations)])
    y = t.cat([t.zeros(len(noun_activations)), t.ones(len(verb_activations)),
               t.zeros(len(adj_activations)), t.zeros(len(other_activations))])
    
    # split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # load into dataloader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=True)

#%%
# define the linear classifier
class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768, 2)

    def forward(self, x):
        return self.linear(x)

cls = LinearClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(cls.parameters(), lr=0.001)
#%%
# train the linear classifier
if MAIN:
    for epoch in range(10):
        for X, y in train_loader:
            optimizer.zero_grad()
            y_pred = cls(X)
            loss = criterion(y_pred, y.long())
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch} loss: {loss.item()}')
#%%
# test the linear classifier
if MAIN:
    cls.eval()
    correct = 0
    total = 0
    with t.no_grad():
        for X, y in test_loader:
            y_pred = cls(X)
            _, predicted = t.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    print(f'Accuracy of the network on the {total:,} test set: {100 * correct / total:.2f}%')
# %%
