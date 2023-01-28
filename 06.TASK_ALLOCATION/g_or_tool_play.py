# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
from datetime import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

from a_config import env_config, ENV_NAME, NUM_TASKS
from c_task_allocation_env import TaskAllocationEnv
from b_task_allocation_with_google_or_tools import solve_by_or_tool


def main(num_episodes):
    env = TaskAllocationEnv(env_config=env_config)
    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        env.reset()

        print("*** GOOGLE OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()

        or_tool_solution = solve_by_or_tool(
            num_tasks=NUM_TASKS, num_resources=2,
            task_demands=env.TASK_RESOURCE_DEMAND,
            resource_capacity=env.INITIAL_RESOURCES_CAPACITY
        )
        or_tool_duration = datetime.now() - or_tool_start_time
        or_tool_solution_lst[i] = or_tool_solution
        or_tool_duration_lst.append(or_tool_duration)
        print()

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
