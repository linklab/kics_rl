# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from a_config import env_config, dqn_config
from c_task_allocation_env import TaskAllocationEnv, ENV_NAME
from e_qnet import QNet, MODEL_DIR
from b_task_allocation_with_google_or_tools import solve_by_or_tool


def play(env, q, num_episodes):
    rl_episode_reward_lst = np.zeros(shape=(num_episodes,), dtype=float)
    rl_duration_lst = []
    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        observation, info = env.reset()
        rl_start_time = datetime.now()
        episode_reward = 0  # cumulative_reward
        episode_steps = 0
        done = False
        print("[EPISODE: {0}]\nRESET, Info: {1}".format(i, info))

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0, action_mask=info["action_mask"])

            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        rl_duration = datetime.now() - rl_start_time
        rl_episode_reward_lst[i] = episode_reward
        rl_duration_lst.append(rl_duration)
        print("*** RL RESULT ***")
        print("EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:5.3f}, INFO:{3}".format(
            i, episode_steps, episode_reward, info
        ))

        print("*** OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()
        or_tool_solution = solve_by_or_tool(
            num_tasks=env.NUM_TASKS,
            num_resources=2,
            task_demands=env.TASK_RESOURCE_DEMAND,
            resource_capacity=env.INITIAL_RESOURCES_CAPACITY
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
        "rl_duration_avg": sum(rl_duration_lst[1:], timedelta(0)) / num_episodes,
        "or_tool_solution_lst": or_tool_solution_lst,
        "or_tool_solutions_avg": np.average(or_tool_solution_lst),
        "or_tool_duration_avg": sum(or_tool_duration_lst[1:], timedelta(0)) / num_episodes
    }


def main_play(num_episodes, env_name):
    env = TaskAllocationEnv(env_config=env_config)

    q = QNet(n_features=(env.NUM_TASKS + 1) * 3, n_actions=env.NUM_TASKS, use_action_mask=dqn_config["use_action_mask"])
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_{0}_{1}_latest.pth".format(env.NUM_TASKS, env_name)))
    q.load_state_dict(model_params)

    results = play(env, q, num_episodes=num_episodes)

    print("[    DQN]   Episode Rewards: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["rl_episode_reward_lst"], results["rl_episode_reward_avg"], results["rl_duration_avg"]
    ))
    print("[OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
