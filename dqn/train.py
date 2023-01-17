import torch
from torch import nn
import matplotlib.pyplot as plt

from config import ENV_SIZE, HIDDEN_DIM, N_EPISODES, N_STEPS, BATCH_SIZE, GAMMA, LR, TAU, LOG_EVERY_N, PATH_MODEL_SAVE
from agent import Agent, Transition
from snake_env import SnakeEnv


if __name__ == '__main__':

    # define agent and environment
    agent = Agent(4, HIDDEN_DIM, 4)
    env = SnakeEnv(ENV_SIZE)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(agent.policy_model.parameters(), lr=LR, amsgrad=True)

    loss_history = []
    reward_history = []

    print('{:>12}  {:>12}  {:>12}'.format('episode', 'loss', 'reward'))
    for episode in range(N_EPISODES):

        loss_per_episode = 0.
        reward_per_episode = 0.

        # generate initial state
        state = torch.tensor(env.get_initial_state(), dtype=torch.float32).unsqueeze(0)

        for step in range(N_STEPS):

            # observation phase
            action_idx = agent.next_action(state)

            next_state, reward = env.step(list(state[0]), action_idx.item())
            next_state, reward = torch.tensor([next_state]), torch.tensor([reward])
            agent.memory.push(Transition(state, action_idx, next_state, reward))

            state = next_state

            # train phase
            if len(agent.memory) >= BATCH_SIZE:
                transitions = agent.memory.sample(BATCH_SIZE)

                state_batch = torch.cat([transition.state for transition in transitions])
                next_state_batch = torch.cat([transition.next_state for transition in transitions])
                action_batch = torch.cat([transition.action_idx for transition in transitions])
                reward_batch = torch.cat([transition.reward for transition in transitions])

                q_value = agent.policy_model(state_batch).gather(1, action_batch)
                next_q_value = agent.target_model(next_state_batch).max(1)[0]

                loss = criterion(q_value.T[0], next_q_value*GAMMA + reward_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_per_episode += loss.item()

            # logging
            reward_per_episode += reward.item()

        # soft update of the target network's weights
        target_net_state_dict = agent.target_model.state_dict()
        policy_net_state_dict = agent.policy_model.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        agent.target_model.load_state_dict(target_net_state_dict)

        # logging
        loss_history.append(round(BATCH_SIZE*loss_per_episode/N_STEPS, 4))
        reward_history.append(round(BATCH_SIZE*reward_per_episode/N_STEPS, 4))

        if episode % LOG_EVERY_N == 0:
            print('{:>12}  {:>12}  {:>12}'.format(episode, loss_history[-1], reward_history[-1]))

    agent.save_model(PATH_MODEL_SAVE + f'model_{loss_history[-1]}.pt')
    print(f'Policy model has been saved as {PATH_MODEL_SAVE + f"model_{loss_history[-1]}.pt"}')

    fig, axs = plt.subplots(1, 2)
    fig.set_figheight(4), fig.set_figwidth(15)

    axs[0].plot(range(N_EPISODES), reward_history), axs[0].set_title('average reward')
    axs[1].plot(range(N_EPISODES), loss_history), axs[1].set_title('average loss')

    plt.show()
