#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from env.aps import CustomAPSEnv
import rl_utils
from models.dqn import DQN
from models.dueling_dqn import Dueling_DQN
from tqdm import tqdm
from motor import MotorController
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Train DQN variants on a custom environment")
    parser.add_argument("--model", type=str, default="DQN", choices=["DQN", "DoubleDQN", "DuelingDQN"],
                        help="Choose the model to train: DQN, DoubleDQN, or DuelingDQN")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--num_episodes", type=int, default=200, help="Number of episodes")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Dimension of hidden layers")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Exploration probability")
    parser.add_argument("--target_update", type=int, default=50, help="Target network update frequency")
    parser.add_argument("--buffer_size", type=int, default=5000, help="Replay buffer size")
    parser.add_argument("--minimal_size", type=int, default=1000, help="Minimum buffer size to start training")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    return parser.parse_args()

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    max_q_value_list = []
    time_list = []
    max_q_value = 0
    time1, time2 = time.time(),time.time()
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                # state = state[0]
                done = False
                while not done:
                    # state = state.to(device)
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995 # 平滑处理
                    max_q_value_list.append(max_q_value) # 保存每个状态的最大Q值
                    action_continuous = rl_utils.dis_to_con(action, env, agent.action_dim)
                    next_state, reward, done, _ = env.step(action_continuous)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                env.action_time.append(time.time()-env.start)
                return_list.append(episode_return)
                time2 = time.time()
                time_list.append(time2-time1)
                time1 = time2
                # if (i_episode+1) % 10 == 0:
                pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % float(np.mean(return_list[-10:]))})
                pbar.update(1)
    motor.loop('b')
    return return_list, max_q_value_list, time_list

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up environment
    env_name = 'APS'
    env = CustomAPSEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = 11  # Discretize continuous actions into 11 discrete actions.

    # Set random seeds for reproducibility
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Initialize replay buffer
    replay_buffer = rl_utils.ReplayBuffer(args.buffer_size)

    # Choose the correct model
    if args.model == "DuelingDQN":
        agent = Dueling_DQN(state_dim, args.hidden_dim, action_dim, args.lr, args.gamma, args.epsilon, args.target_update, device, 'DuelingDQN')
    elif args.model == "DoubleDQN":
        agent = DQN(state_dim, args.hidden_dim, action_dim, args.lr, args.gamma, args.epsilon, args.target_update, device, 'DoubleDQN')
    else:
        agent = DQN(state_dim, args.hidden_dim, action_dim, args.lr, args.gamma, args.epsilon, args.target_update, device)

    # Train the model
    return_list, max_q_value_list, time_list = train_DQN(agent, env, args.num_episodes, replay_buffer, args.minimal_size, args.batch_size)

    # Plot results
    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{args.model} on {env_name}')
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title(f'{args.model} on {env_name}')
    plt.show()

    # Save data
    with open(f'./data_pickle/{args.model.lower()}_rt_maxq_time.pkl', 'wb') as f:
        pickle.dump((return_list, max_q_value_list, time_list), f)

if __name__ == "__main__":
    main()
