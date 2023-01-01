import json
import argparse

import torch

from config import NUMB_SAMPLES, K_CANDIDATES, VOCAB_PATH, \
                   LOAD_MODEL_PATH, DEVICE

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_tokens', default=' ', help='')
    args = parser.parse_args()

    with open(VOCAB_PATH, 'r') as f:
        json_data = json.load(f)

    idx2token = {int(k):v for k,v in json_data.items()}
    token2idx = {v: k for k, v in idx2token.items()}

    model = torch.load(LOAD_MODEL_PATH)
    model.eval()

    start_tokens = args.start_tokens

    for n in range(NUMB_SAMPLES):
        idx = [token2idx[token.lower()] for token in list(start_tokens)]
        idx = torch.tensor([idx]).to(DEVICE)
        result = model.generate(idx, idx2token, K_CANDIDATES)
        print(result)
