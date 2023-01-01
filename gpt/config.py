# model config
EMB_DIM = 32
MAX_LENGTH = 12
N_LAYERS = 4
N_HEADS = 2
DROPOUT = 0.1

# training config
WEIGHT_DECAY = 0.001
BATCH_SIZE = 32
N_EPOCHS = 3000
LR = 0.0001
WEIGHT_DECAY = 0.
LOG_EVERY_N = 500
DEVICE = 'cpu'

# training paths
DATA_PATH = 'names.txt'
SAVE_MODEL_PATH = ''

# generation config
NUMB_SAMPLES = 5  # a number of samples to generate
K_CANDIDATES = 3  # a number of token candidates to take while generating a sequence

# generation paths
LOAD_MODEL_PATH = ""  # path to checkpoint (e.g. checkpoint_0.832.pt)
VOCAB_PATH = 'vocab.json'
