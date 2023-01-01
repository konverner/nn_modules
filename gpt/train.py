import math

import torch 
import torch.nn.functional as F

from config import EMB_DIM, MAX_LENGTH, N_LAYERS, N_HEADS, \
                    DROPOUT, WEIGHT_DECAY, BATCH_SIZE, N_EPOCHS, \
                    LR, WEIGHT_DECAY, DATA_PAT, LOG_EVERY_N, \
                    SAVE_MODEL_PATH, DEVICE


class Dataset:
  def __init__(self, X, Y):
    self.X = X
    self.Y = Y

  def __len__(self):
    return len(self.X)
    
  def __getitem__(self, index):
    return self.X[index], self.Y[index]
  
if __name__ == "__main__":
  with open(DATA_PATH, 'r') as f:
    names = [line.lower() for line in f.read().split('\n')]

  # create vocabulary of tokens (letters) with padding token 
  vocab = ['[PAD]'] + sorted(list(set(''.join(names))))
  idx2token = {k:v for k,v in enumerate(vocab)}
  token2idx = {v:k for k,v in idx2token.items()}
  print(len(vocab))

  # create input and target by shifting the text by one letter
  # e.g. "river" -> "river", "iver[PAD]" 
  X, Y = [], []
  for name in names:
    tokens = list(map(token2idx.get, name))
    while len(tokens) < MAX_LENGTH:
      tokens.append(token2idx['[PAD]'])
    x, y = tokens, tokens[1:] + [token2idx['[PAD]']]
    X.append(torch.tensor(x))
    Y.append(torch.tensor(y))

  N = len(X)
  VOCAB_SIZE = len(vocab)

  dataset = Dataset(X, Y)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
  
  model = GPT(VOCAB_SIZE, EMB_DIM, N_LAYERS, N_HEADS, 512, dropout=DROPOUT, max_length=MAX_LENGTH, device=DEVICE)
  
  print(model.count_parameters())
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

  for epoch in range(N_EPOCHS):
    epoch_loss_value = 0.
    for x, y in dataloader:
      x, y = x.to(DEVICE), y.to(DEVICE)
      probs = model(x)
      loss_value = F.cross_entropy(probs.view(-1, probs.size(-1)), y.view(-1), ignore_index=0)
      loss_value.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss_value += loss_value.item()

    if epoch % LOG_EVERY_N == 0:
      print(epoch_loss_value/len(dataloader))
 torch.save(model, f'{SAVE_MODEL_PATH}checkpoint_{average_loss}.pt')
 print(f'model saved as {SAVE_MODEL_PATH}checkpoint_{average_loss}.pt')
