# Generative Pre-trained Transformer

Here is implementation in PyTorch of Generative Pre-trained Transformer (GPT) from the paper "Improving Language Understanding by Generative Pre-Training." 
The implementation is not fully optimized to have great perfomance since the main purpose of the code is educative. The code is provided with test on a simple
name generation task accomplished on `names.txt` data.

# Get started

## Train

First of all, adjust configuration in `config.py`. Secondly, prepare a dataset that contains one piece of text data per a line (see `names.txt`). Thirdly,
run `train.py` script in a line command.

## Generation

Firstly, provide model weights in (`*.pt`), vocabulary (`vocab.json`) and adjust other configurations if needed in `config.py` file. Secondly, run `generate.py` script in
a line command providing a string to start with; to generate samples with random initial letters, do not provide any string.

For example: having trained on `names.txt` model:

`!python /content/nn_modules/gpt/generate.py`

```
harris
mccormick
harmon
marsh
barrera
```

with starting string:

`!python /content/nn_modules/gpt/generate.py ze`

```
zeloce
zelexan
zelene
zery
zelexer
```
