import os
from datetime import datetime, timedelta

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

from a_config import env_config, ENV_NAME, NUM_TASKS
from c_task_allocation_env import TaskAllocationEnv
from b_task_allocation_with_google_or_tools import solve


def main(num_episodes):
    env = TaskAllocationEnv(env_config=env_config)
    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        env.reset()

        print("*** GOOGLE OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()

        or_tool_solution = solve(
            num_tasks=NUM_TASKS, num_resources=2,
            task_values=env.TASK_VALUES,
            task_demands=env.TASK_RESOURCE_DEMAND,
            resource_capacity=env.INITIAL_RESOURCES_CAPACITY
        )
        or_tool_duration = datetime.now() - or_tool_start_time
        or_tool_solution_lst[i] = or_tool_solution
        or_tool_duration_lst.append(or_tool_duration)
        print()

    results = {
        "or_tool_solution_lst": or_tool_solution_lst,
        "or_tool_solutions_avg": np.average(or_tool_solution_lst),
        "or_tool_duration_avg": sum(or_tool_duration_lst[1:], timedelta(0)) / (num_episodes - 1)
    }

    print("[OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES)
