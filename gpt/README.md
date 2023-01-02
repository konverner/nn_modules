## Generative Pre-trained Transformer

This is PyTorch implementation of Generative Pre-trained Transformer (GPT) from the paper "Improving Language Understanding by Generative Pre-Training." 
The implementation is not fully optimized to have great perfomance since the main purpose of the code is educative. The code is provided with test on a simple
name generation task accomplished on `names.txt` data.

Detailed explanation of GPT theory and this implementation is avaliable on [Medium](https://medium.com/@konst.verner/gpt-explanation-and-implementation-from-scratch-in-pytorch-9962839417ac) and interective version of the code with an example can be accessed via [Colab Notebook](https://colab.research.google.com/drive/1bxMkHVbRP0NRkEPU1FCadisDF3b8XP49?usp=sharing).  

## Get started

**Train**

1) adjust configuration in `config.py`

2) prepare a dataset that contains one piece of text data per line (see `names.txt`)

3) run `train.py` script in a line command

**Generation**

1) provide paths to model weights in (`*.pt`), vocabulary (`vocab.json`) and adjust other configurations if needed in `config.py` file

2) run `generate.py` script in a line command providing a string to start with; to generate samples with random initial letters, do not provide any string.

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
