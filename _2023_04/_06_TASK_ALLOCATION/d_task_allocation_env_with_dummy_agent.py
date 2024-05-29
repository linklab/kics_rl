import random

import numpy as np

from c_task_allocation_env import TaskAllocationEnv
from a_config import env_config

class Dummy_Agent:
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks

    def get_action(self, observation, action_mask):
        # observation is not used
        available_actions = np.where(action_mask == 0)[0]
        action_id = random.choice(available_actions)
        return action_id


def main():
    print("START RUN!!!")
    # env_config = {
    #     "num_tasks": 10,  # 대기하는 태스크 개수
    #     "use_static_task_resource_demand": False,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
    #     "use_same_task_resource_demand": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
    #     "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
    #     "high_demand_resource_at_task": [100, 100],  # 태스크의 각 자원 최대 요구량
    #     "initial_resources_capacity": [250, 250],  # 초기 자원 용량
    # }
    env = TaskAllocationEnv(env_config=env_config)

    agent = Dummy_Agent(env.NUM_TASKS)
    observation, info = env.reset()

    episode_step = 0
    done = False
    print("[Step: RESET] Info: {0}".format(info))
    print(info['INTERNAL_STATE'].flatten())

    while not done:
        action = agent.get_action(observation, info["ACTION_MASK"])
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_step += 1
        print("[Step: {0:3}] Obs.: {1}, Action: {2:>2}, Next Obs.: {3}, "
              "Reward: {4:>6.3f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
            episode_step, observation.shape, action, next_observation.shape,
            reward, terminated, truncated, info
        ))
        print(info['INTERNAL_STATE'].flatten())
        observation = next_observation
        done = terminated or truncated


if __name__ == "__main__":
    main()
