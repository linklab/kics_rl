# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time

import gymnasium as gym
import torch
from wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from q_net import QNetwork, MODEL_DIR
from train import args
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="human")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


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
    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"

    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name)]
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
