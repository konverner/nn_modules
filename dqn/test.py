import torch
import matplotlib.pyplot as plt

from agent import Agent
from snake_env import SnakeEnv
from config import ENV_SIZE, HIDDEN_DIM, PATH_MODEL_LOAD

agent = Agent(4, HIDDEN_DIM, 4)
agent.load_model(PATH_MODEL_LOAD)

env = SnakeEnv(ENV_SIZE)

state = torch.tensor(env.get_initial_state(), dtype=torch.float32).unsqueeze(0)
total_reward = 0.
for i in range(30):
  env.render(list(state[0].int().numpy()))
  plt.draw()
  plt.pause(0.1)
  plt.close()

  action = agent.next_action(state)
  new_state, reward = env.step(list(state[0]), action.item())
  total_reward += reward
  new_state, reward = torch.tensor([new_state]), torch.tensor([reward])
  state = new_state

print(f'total reward: {total_reward}')
