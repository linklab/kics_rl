import random

import numpy as np

from b_task_allocation_env import TaskAllocationEnv


class Dummy_Agent:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks

    def get_action(self, observation, action_mask):
        # observation is not used
        available_actions = np.where(action_mask == 1.0)[0]
        action_id = random.choice(available_actions)
        return action_id


def main():
    print("START RUN!!!")
    env = TaskAllocationEnv()
    agent = Dummy_Agent(env.NUM_TASKS)
    observation, info = env.reset()

    episode_step = 0
    done = False

    while not done:
        action = agent.get_action(observation, info["action_mask"])
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_step += 1
        print("[Step: {0:3}] Obs.: {1}, Action: {2:>2}, Next Obs.: {3}, "
              "Reward: {4:>6.3f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
            episode_step, observation.shape, action, next_observation.shape,
            reward, terminated, truncated, info["action_mask"]
        ))
        observation = next_observation
        done = terminated or truncated


if __name__ == "__main__":
    main()
