# Problem: Multiple Tasks Allocation to One Computing server
import gymnasium as gym
import numpy as np
import copy
import enum

from gymnasium import spaces

from a_config import STATIC_TASK_RESOURCE_DEMAND_SAMPLE, STATIC_TASK_VALUE_SAMPLE
import random
from datetime import datetime

class DoneReasonType(enum.Enum):
    TYPE_FAIL_1 = "The Same Task Selected"
    TYPE_FAIL_2 = "Resource Limit Exceeded"
    TYPE_SUCCESS_1 = "All Tasks Allocated Successfully"
    TYPE_SUCCESS_2 = "An Unavailable Resource"


class TaskAllocationEnv(gym.Env):
    def __init__(self, env_config):
        super(TaskAllocationEnv, self).__init__()

        random.seed(datetime.now().timestamp())

        self.internal_state = None
        self.actions_selected = None

        self.cpu_allocated = None
        self.ram_allocated = None
        self.cpu_ram_allocated = None
        self.value_allocated = None

        self.resources_step = None

        self.total_value = None
        self.min_task_cpu_demand = None
        self.min_task_ram_demand = None
        self.total_cpu_demand = None
        self.total_ram_demand = None
        self.total_resource_demand = None
        self.value_step = None

        self.action_mask = None

        self.NUM_TASKS = env_config["num_tasks"]
        self.INITIAL_RESOURCES_CAPACITY = env_config["initial_resources_capacity"]
        self.MIN_RESOURCE_DEMAND_AT_TASK = env_config["low_demand_resource_at_task"]
        self.MAX_RESOURCE_DEMAND_AT_TASK = env_config["high_demand_resource_at_task"]
        self.MIN_VALUE_AT_TASK = env_config["low_value_at_task"]
        self.MAX_VALUE_AT_TASK = env_config["high_value_at_task"]
        self.USE_STATIC_TASK_RESOURCE_DEMAND = env_config["use_static_task_resource_demand"]
        self.USE_SAME_TASK_RESOURCE_DEMAND = env_config["use_same_task_resource_demand"]

        self.action_space = spaces.Discrete(n=self.NUM_TASKS)
        self.observation_space = spaces.Discrete(n=(self.NUM_TASKS + 1) * 4)

        self.CPU_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[0]
        self.RAM_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[1]
        self.TOTAL_RESOURCE_CAPACITY = sum(self.INITIAL_RESOURCES_CAPACITY)

        self.TASK_RESOURCE_DEMAND = None
        self.TASK_VALUES = None

        print("{0:>50}: {1}".format("NUM_TASKS", self.NUM_TASKS))
        print("{0:>50}: {1}".format(
            "LOW_DEMAND_RESOURCE_AT_TASK", env_config["low_demand_resource_at_task"]
        ))
        print("{0:>50}: {1}".format(
            "HIGH_DEMAND_RESOURCE_AT_TASK", env_config["high_demand_resource_at_task"]
        ))
        print("{0:>50}: {1}".format(
            "LOW_VALUE_AT_TASK", env_config["low_value_at_task"]
        ))
        print("{0:>50}: {1}".format(
            "HIGH_VALUE_AT_TASK", env_config["high_value_at_task"]
        ))
        print("{0:>50}: {1}".format(
            "INITIAL_RESOURCES_CAPACITY", env_config["initial_resources_capacity"]
        ))
        print("{0:>50}: {1}".format(
            "USE_STATIC_TASK_RESOURCE_DEMAND", env_config["use_static_task_resource_demand"]
        ))
        print("{0:>50}: {1}".format(
            "USE_SAME_TASK_RESOURCE_DEMAND", env_config["use_same_task_resource_demand"]
        ))

    def get_initial_internal_state(self):
        state = np.zeros(shape=(self.NUM_TASKS + 1, 4), dtype=float)

        if self.USE_STATIC_TASK_RESOURCE_DEMAND:
            self.TASK_RESOURCE_DEMAND = STATIC_TASK_RESOURCE_DEMAND_SAMPLE
            self.TASK_VALUES = STATIC_TASK_VALUE_SAMPLE
        else:
            if self.USE_SAME_TASK_RESOURCE_DEMAND:
                if self.TASK_RESOURCE_DEMAND is None:
                    self.TASK_RESOURCE_DEMAND = np.zeros(shape=(self.NUM_TASKS, 2))
                    self.TASK_VALUES = np.zeros(shape=(self.NUM_TASKS,))
                    for task_idx in range(self.NUM_TASKS):
                        self.TASK_RESOURCE_DEMAND[task_idx] = np.random.randint(
                            low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(2,)
                        )
                        self.TASK_VALUES[task_idx] = np.random.randint(
                            low=self.MIN_VALUE_AT_TASK, high=self.MAX_VALUE_AT_TASK, size=(1,)
                        )
                    # self.TASK_RESOURCE_DEMAND = np.sort(self.TASK_RESOURCE_DEMAND, axis=0)
            else:
                self.TASK_RESOURCE_DEMAND = np.zeros(shape=(self.NUM_TASKS, 2))
                self.TASK_VALUES = np.zeros(shape=(self.NUM_TASKS,))
                for task_idx in range(self.NUM_TASKS):
                    self.TASK_RESOURCE_DEMAND[task_idx] = np.random.randint(
                        low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(2, )
                    )
                    self.TASK_VALUES[task_idx] = np.random.randint(
                        low=self.MIN_VALUE_AT_TASK, high=self.MAX_VALUE_AT_TASK, size=(1,)
                    )
                # self.TASK_RESOURCE_DEMAND = np.sort(self.TASK_RESOURCE_DEMAND, axis=0)

        state[:-1, 2:] = self.TASK_RESOURCE_DEMAND
        state[:-1, 1] = self.TASK_VALUES

        self.total_value = state[:-1, 1].sum()

        self.min_task_cpu_demand = state[:-1, 2].min()
        self.min_task_ram_demand = state[:-1, 3].min()
        self.total_cpu_demand = state[:-1, 2].sum()
        self.total_ram_demand = state[:-1, 3].sum()
        self.total_resource_demand = self.total_cpu_demand + self.total_ram_demand

        state[-1][2:] = np.array(self.INITIAL_RESOURCES_CAPACITY)

        return state

    @staticmethod
    def get_observation_from_internal_state(internal_state):
        observation = internal_state.flatten()
        return observation

    def reset(self, **kwargs):
        self.internal_state = self.get_initial_internal_state()
        self.actions_selected = []
        self.cpu_ram_allocated = 0
        self.cpu_allocated = 0
        self.ram_allocated = 0
        self.value_allocated = 0
        self.action_mask = np.zeros(shape=(self.NUM_TASKS,), dtype=float)

        observation = self.internal_state.flatten()

        info = {}
        self.fill_info(info)
        info["ACTION_MASK"] = self.action_mask

        return observation, info

    def step(self, action_idx):
        info = {}
        self.actions_selected.append(action_idx)

        self.value_step = self.internal_state[action_idx][1]
        cpu_step = self.internal_state[action_idx][2]
        ram_step = self.internal_state[action_idx][3]
        self.resources_step = cpu_step + ram_step

        assert (self.internal_state[action_idx][0] == 0), "The Same Task Selected: {0}".format(action_idx)
        assert (self.cpu_allocated + cpu_step <= self.CPU_RESOURCE_CAPACITY) and \
                    (self.ram_allocated + ram_step <= self.RAM_RESOURCE_CAPACITY), "Resource Limit Exceeded"

        terminated = False

        self.internal_state[action_idx][0] = 1.0
        self.internal_state[action_idx][1] = 0.0
        self.internal_state[action_idx][2] = 0.0
        self.internal_state[action_idx][3] = 0.0

        self.internal_state[-1][1] = self.internal_state[-1][1] + self.value_step
        self.internal_state[-1][2] = self.internal_state[-1][2] - cpu_step
        self.internal_state[-1][3] = self.internal_state[-1][3] - ram_step

        self.cpu_allocated = self.cpu_allocated + cpu_step
        self.ram_allocated = self.ram_allocated + ram_step
        self.cpu_ram_allocated = self.cpu_ram_allocated + cpu_step + ram_step
        self.value_allocated = self.value_allocated + self.value_step

        unavailable_tasks = np.where(
            (self.internal_state[:-1, 2] == 0.0) |                          # CPU를 이미 할당을 했거나
            (self.internal_state[:-1, 2] > self.internal_state[-1][2]) |    # CPU Demand가 남아 있는 CPU 자원보다 크거나
            (self.internal_state[:-1, 3] == 0.0) |                          # RAM을 이미 할당을 했거나
            (self.internal_state[:-1, 3] > self.internal_state[-1][3])      # RAM Demand가 남아 있는 RAM 자원보다 크거나
        )[0]

        if len(unavailable_tasks) == self.NUM_TASKS:
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_2
        else:
            self.action_mask[unavailable_tasks] = 1.0

        next_observation = self.internal_state.flatten()

        self.fill_info(info)

        if terminated:
            info["ACTION_MASK"] = np.ones(shape=(self.NUM_TASKS,), dtype=float) * -1.0
        else:
            info["ACTION_MASK"] = self.action_mask

        reward = self.get_reward()

        truncated = None

        return next_observation, reward, terminated, truncated, info

    def get_reward(self):
        reward = self.value_step / self.total_value
        assert reward < 1.0
        return reward

    def fill_info(self, info):
        info["TOTAL_RESOURCE_CAPACITY"] = self.TOTAL_RESOURCE_CAPACITY
        info["CPU_CAPACITY"] = self.CPU_RESOURCE_CAPACITY
        info["RAM_CAPACITY"] = self.RAM_RESOURCE_CAPACITY
        info["TOTAL_RESOURCE_DEMAND"] = self.total_resource_demand
        info["TOTAL_CPU_DEMAND"] = self.total_cpu_demand
        info["TOTAL_RAM_DEMAND"] = self.total_ram_demand
        info["ACTIONS_SELECTED"] = self.actions_selected
        info["TOTAL_ALLOCATED"] = self.cpu_ram_allocated
        info["CPU_ALLOCATED"] = self.cpu_allocated
        info["RAM_ALLOCATED"] = self.ram_allocated
        info["INTERNAL_STATE"] = self.internal_state
        info["VALUE_ALLOCATED"] = self.value_allocated

