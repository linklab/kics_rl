# Problem: Multiple Tasks Allocation to One Computing server
import operator
import random

import gymnasium as gym
import numpy as np
import copy
import enum

ENV_NAME = "Task_Allocation"

# env_config = {
#     "num_tasks": 10,  # 대기하는 태스크 개수
#     "use_static_task_resource_demand": True,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
#     "same_task_resource_demand": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
#     "initial_resources_capacity": [100, 100],  # 초기 자원 용량
#     "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
#     "high_demand_resource_at_task": [20, 20]  # 태스크의 각 자원 최대 요구량
# }

env_config = {
    "num_tasks": 3,  # 대기하는 태스크 개수
    "use_static_task_resource_demand": False,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
    "same_task_resource_demand": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
    "initial_resources_capacity": [30, 30],  # 초기 자원 용량
    "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
    "high_demand_resource_at_task": [20, 20]  # 태스크의 각 자원 최대 요구량
}

if env_config["same_task_resource_demand"]:
    assert env_config["use_static_task_resource_demand"] is False

if env_config["use_static_task_resource_demand"]:
    assert env_config["num_tasks"] == 10


class DoneReasonType(enum.Enum):
    TYPE_FAIL_1 = "The Same Task Selected"
    TYPE_FAIL_2 = "Resource Limit Exceeded"
    TYPE_SUCCESS_1 = "All Tasks Allocated Successfully - [ALL]"
    TYPE_SUCCESS_2 = "All Resources Used Up - [BEST]"
    TYPE_SUCCESS_3 = "A Resource Used Up - [GOOD]"


class TaskAllocationEnv(gym.Env):
    def __init__(self):
        super(TaskAllocationEnv, self).__init__()

        self.internal_state = None
        self.actions_selected = None
        self.total_allocated = None

        self.cpu_allocated = None
        self.ram_allocated = None
        self.total_allocated = None

        self.resources_step = None

        self.min_task_cpu_demand = None
        self.min_task_ram_demand = None
        self.total_cpu_demand = None
        self.total_ram_demand = None
        self.total_resource_demand = None

        self.action_mask = None

        self.NUM_TASKS = env_config["num_tasks"]
        self.INITIAL_RESOURCES_CAPACITY = env_config["initial_resources_capacity"]
        self.MIN_RESOURCE_DEMAND_AT_TASK = env_config["low_demand_resource_at_task"]
        self.MAX_RESOURCE_DEMAND_AT_TASK = env_config["high_demand_resource_at_task"]
        self.USE_STATIC_TASK_RESOURCE_DEMAND = env_config["use_static_task_resource_demand"]
        self.SAME_TASK_RESOURCE_DEMAND = env_config["same_task_resource_demand"]

        self.CPU_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[0]
        self.RAM_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[1]
        self.TOTAL_RESOURCE_CAPACITY = sum(self.INITIAL_RESOURCES_CAPACITY)

        self.STATIC_TASK_RESOURCE_DEMAND = [
            [6, 12],
            [4, 17],
            [8, 14],
            [14, 7],
            [14, 9],
            [13, 12],
            [12, 14],
            [17, 10],
            [12, 15],
            [13, 14],
        ]
        
        self.TASK_RESOURCE_DEMAND = None

        print("NUM_TASKS:", self.NUM_TASKS)
        print("USE_STATIC_TASK_RESOURCE_DEMAND:", self.USE_STATIC_TASK_RESOURCE_DEMAND)
        print("SAME_TASK_RESOURCE_DEMAND:", self.SAME_TASK_RESOURCE_DEMAND)
        print("###########################################################")

    def get_initial_internal_state(self):
        state = np.zeros(shape=(self.NUM_TASKS + 1, 3), dtype=int)

        if self.USE_STATIC_TASK_RESOURCE_DEMAND is True:
            state[:-1, 1:] = self.STATIC_TASK_RESOURCE_DEMAND
        else:
            if self.SAME_TASK_RESOURCE_DEMAND:
                if self.TASK_RESOURCE_DEMAND is None:
                    self.TASK_RESOURCE_DEMAND = np.zeros(shape=(self.NUM_TASKS, 2))
                    for task_idx in range(self.NUM_TASKS):
                        self.TASK_RESOURCE_DEMAND[task_idx] = np.random.randint(
                            low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(2, )
                        )
                    self.TASK_RESOURCE_DEMAND = np.sort(self.TASK_RESOURCE_DEMAND, axis=0)

                state[:-1, 1:] = self.TASK_RESOURCE_DEMAND
            else:
                self.TASK_RESOURCE_DEMAND = np.zeros(shape=(self.NUM_TASKS, 2))
                for task_idx in range(self.NUM_TASKS):
                    self.TASK_RESOURCE_DEMAND[task_idx] = np.random.randint(
                        low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(2, )
                    )
                self.TASK_RESOURCE_DEMAND = np.sort(self.TASK_RESOURCE_DEMAND, axis=0)
                state[:-1, 1:] = self.TASK_RESOURCE_DEMAND

        self.min_task_cpu_demand = state[:-1, 1].min()
        self.min_task_ram_demand = state[:-1, 2].min()
        self.total_cpu_demand = state[:-1, 1].sum()
        self.total_ram_demand = state[:-1, 2].sum()
        self.total_resource_demand = self.total_cpu_demand + self.total_ram_demand

        state[-1][1:] = np.array(self.INITIAL_RESOURCES_CAPACITY)

        return state

    def get_observation_from_internal_state(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.TOTAL_RESOURCE_CAPACITY
        return observation

    def reset(self, **kwargs):
        self.internal_state = self.get_initial_internal_state()
        self.actions_selected = []
        self.total_allocated = 0
        self.cpu_allocated = 0
        self.ram_allocated = 0
        self.action_mask = np.zeros(shape=(self.NUM_TASKS,), dtype=float)

        observation = self.get_observation_from_internal_state()
        info = {
            "action_mask": self.action_mask
        }

        return observation, info

    def step(self, action_idx):
        info = {}
        self.actions_selected.append(action_idx)

        self.action_mask[action_idx] = 1.0

        cpu_step = self.internal_state[action_idx][1]
        ram_step = self.internal_state[action_idx][2]

        self.resources_step = cpu_step + ram_step

        ###########################
        ### terminated 결정 - 시작 ###
        ###########################
        terminated = False
        if self.internal_state[action_idx][0] == 1:
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_FAIL_1   ##### [TYPE 1] The Same Task Selected #####

        elif (self.cpu_allocated + cpu_step > self.CPU_RESOURCE_CAPACITY) or \
                    (self.ram_allocated + ram_step > self.RAM_RESOURCE_CAPACITY):
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_FAIL_2   ##### [TYPE 2] Resource Limit Exceeded #####

        else:
            self.internal_state[action_idx][0] = 1
            self.internal_state[action_idx][1] = -1
            self.internal_state[action_idx][2] = -1

            self.internal_state[-1][1] = self.internal_state[-1][1] - cpu_step
            self.internal_state[-1][2] = self.internal_state[-1][2] - ram_step

            self.cpu_allocated = self.cpu_allocated + cpu_step
            self.ram_allocated = self.ram_allocated + ram_step
            self.total_allocated = self.total_allocated + cpu_step + ram_step

            conditions = [
                self.internal_state[-1][1] < self.min_task_cpu_demand,
                self.internal_state[-1][2] < self.min_task_ram_demand
            ]

            if 0 not in self.internal_state[:self.NUM_TASKS, 0]:
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_1  ##### All Tasks Allocated Successfully - [ALL] #####
            elif all(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_2  ##### All Resources Used Up - [BEST] #####
            elif any(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_3  ##### A Resource Used Up - [GOOD] #####
            else:
                pass
        ###########################
        ### terminated 결정 - 종료 ###
        ###########################

        new_observation = self.get_observation_from_internal_state()

        if terminated:
            reward = self.get_reward(done_type=info['DoneReasonType'])
        else:
            reward = self.get_reward(done_type=None)

        info["TOTAL_RESOURCE_CAPACITY"] = self.TOTAL_RESOURCE_CAPACITY
        info["CPU_CAPACITY"] = self.CPU_RESOURCE_CAPACITY
        info["RAM_CAPACITY"] = self.RAM_RESOURCE_CAPACITY
        info["TOTAL_RESOURCE_DEMAND"] = self.total_resource_demand
        info["TOTAL_CPU_DEMAND"] = self.total_cpu_demand
        info["TOTAL_RAM_DEMAND"] = self.total_ram_demand
        info["ACTIONS_SELECTED"] = self.actions_selected
        info["TOTAL_ALLOCATED"] = self.total_allocated
        info["CPU_ALLOCATED"] = self.cpu_allocated
        info["RAM_ALLOCATED"] = self.ram_allocated
        info["INTERNAL_STATE"] = self.internal_state
        info["action_mask"] = self.action_mask

        truncated = None

        return new_observation, reward, terminated, truncated, info

    def get_reward(self, done_type=None):
        # The Same Task Selected or
        # Resource Limit Exceeded
        if done_type == DoneReasonType.TYPE_FAIL_1 or done_type == DoneReasonType.TYPE_FAIL_2:
            fail_reward = -1.0
        else:
            fail_reward = 0.0

        reward = self.resources_step / self.TOTAL_RESOURCE_CAPACITY

        return reward + fail_reward
