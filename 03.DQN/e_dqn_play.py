# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import torch

from c_qnet import QNet, DEVICE, MODEL_DIR

ENV_NAME = "CartPole-v1"


def play(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_q_play(num_episodes):
    env = gym.make(ENV_NAME, render_mode="human")

    q = QNet(n_features=4, n_actions=2)
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_CartPole-v1_latest.pth"))
    q.load_state_dict(model_params)

    play(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    main_q_play(num_episodes=NUM_EPISODES)
