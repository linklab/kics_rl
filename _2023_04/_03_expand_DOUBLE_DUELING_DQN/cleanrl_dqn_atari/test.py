# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time

import gymnasium as gym
import torch

from q_net import QNetwork, MODEL_DIR
from train import args, make_env
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(torch.Tensor(observation).to(DEVICE), epsilon=0.0)

            next_observation, reward, terminated, truncated, infos = env.step(action)

            episode_reward += reward.squeeze(-1)
            observation = next_observation
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        done = True

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


def main_play(num_episodes, env_name):
    args.test = True

    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.test)]
    )

    q = QNetwork(env).to(DEVICE)
    model_params = torch.load(os.path.join(MODEL_DIR, "double_dueling_dqn_{0}_latest.pth".format(env_name)))
    q.load_state_dict(model_params)

    test(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "BreakoutNoFrameskip-v4"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
