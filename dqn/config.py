# environment
ENV_SIZE = (5, 5)
ACTIONS = [[0, 1], [1, 0], [0, -1], [-1, 0]]
N_STATES, N_ACTIONS = ENV_SIZE[0]*ENV_SIZE[1], len(ACTIONS)

# learning
HIDDEN_DIM = 128
BATCH_SIZE = 4
N_EPISODES = 1000
N_STEPS = 32  # number of step per episode
TAU = 0.05  # coefficient for soft update of the target network's weights
GAMMA = 0.9
LR = 0.0004
LOG_EVERY_N = 100  # logging every Nth episode

# paths
PATH_MODEL_SAVE = ''  # where to save a model after training
PATH_MODEL_LOAD = ''  # where to load a model for testing from
