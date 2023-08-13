import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from datetime import datetime, timedelta
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from _2023_08._01_DQN_MKP.a_config import env_config, ENV_NAME, NUM_ITEMS, NUM_RESOURCES
from _2023_08._01_DQN_MKP.c_mkp_env import MkpEnv
from _2023_08._02_DQN_ATTN_MKP.c_dqn_attn_train import QNetAttn
from _2023_08._01_DQN_MKP.b_mkp_with_google_or_tools import solve

DEVICE = torch.device("cpu")


def test(env, q, num_episodes):
    rl_episode_reward_lst = np.zeros(shape=(num_episodes,), dtype=float)
    rl_duration_lst = []
    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0  # cumulative_reward
        episode_steps = 0
        done = False
        print("[EPISODE: {0}]\nRESET, Info: {1}".format(i, info))

        rl_start_time = datetime.now()
        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0, action_mask=info["ACTION_MASK"])

            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        rl_duration = datetime.now() - rl_start_time
        rl_episode_reward_lst[i] = info["VALUE_ALLOCATED"]
        rl_duration_lst.append(rl_duration)
        print("*** RL RESULT ***")
        print("EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:5.3f}, INFO:{3}".format(
            i, episode_steps, episode_reward, info
        ))

        print("*** GOOGLE OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()

        or_tool_solution = solve(
            n_items=NUM_ITEMS, n_resources=2,
            item_resource_demands=env.item_resource_demand,
            item_values=env.item_values,
            resource_capacities=env.initial_resources_capacity
        )
        or_tool_duration = datetime.now() - or_tool_start_time
        or_tool_solution_lst[i] = or_tool_solution
        or_tool_duration_lst.append(or_tool_duration)

        print("*** RL VS. OR_TOOL COMPARISON ***")
        print("RL_EPISODE_REWARD (UTILIZATION) | OR_TOOL_SOLUTION (UTILIZATION) ---> {0:>6.3f} |{1:>6.3f}".format(
            episode_reward, or_tool_solution
        ))

        print("RL_DURATION | OR_TOOL_DURATION ---> {0} | {1}".format(
            rl_duration, or_tool_duration,
        ))

        print()

    return {
        "rl_episode_reward_lst": rl_episode_reward_lst,
        "rl_episode_reward_avg": np.average(rl_episode_reward_lst),
        "rl_duration_avg": sum(rl_duration_lst[1:], timedelta(0)) / (num_episodes -1),
        "or_tool_solution_lst": or_tool_solution_lst,
        "or_tool_solutions_avg": np.average(or_tool_solution_lst),
        "or_tool_duration_avg": sum(or_tool_duration_lst[1:], timedelta(0)) / (num_episodes - 1)
    }


def main(num_episodes, env_name):
    current_path = os.path.dirname(os.path.realpath(__file__))
    project_home = os.path.abspath(os.path.join(current_path, os.pardir))
    model_dir = os.path.join(project_home, "_02_DQN_ATTN_MKP", "models")

    env = MkpEnv(env_config=env_config)

    print("*" * 100)

    q = QNetAttn(n_features=NUM_ITEMS * (NUM_RESOURCES + 1), n_actions=NUM_ITEMS, device=DEVICE)

    model_params = torch.load(
        os.path.join(model_dir, "dqn_{0}_{1}_latest.pth".format(NUM_ITEMS, env_name))
    )
    q.load_state_dict(model_params)

    results = test(env, q, num_episodes=num_episodes)

    print("[    DQN]   Episode Rewards: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["rl_episode_reward_lst"], results["rl_episode_reward_avg"], results["rl_duration_avg"]
    ))
    print("[OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
