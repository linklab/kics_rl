# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.2f" % x))

import torch
from a_task_allocation_env import TaskAllocationEnv, ENV_NAME
from b_qnet import QNet, MODEL_DIR


def play(env, q, num_episodes):
    episode_reward_lst = np.zeros(shape=(num_episodes,), dtype=float)

    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        episode_reward_lst[i] = episode_reward

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:5.3f}, INFO:{3}".format(
            i, episode_steps, episode_reward, info
        ))

    return episode_reward_lst, np.average(episode_reward_lst)


def main_play(num_episodes, env_name):
    env = TaskAllocationEnv()

    q = QNet(n_features=(env.NUM_TASKS + 1) * 3, n_actions=env.NUM_TASKS)
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_{0}_{1}_latest.pth".format(env.NUM_TASKS, env_name)))
    q.load_state_dict(model_params)

    episode_reward_lst, episode_reward_avg = play(env, q, num_episodes=num_episodes)
    print("[Play Episode Reward: {0}] Average: {1:.3f}".format(episode_reward_lst, episode_reward_avg))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
