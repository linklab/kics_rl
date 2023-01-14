# Problem: Multiple Tasks Allocation to One Computing server
import gymnasium as gym
import numpy as np
import copy
import enum

ENV_NAME = "Task_Allocation"

env_config = {
    "num_tasks": 7,  # 대기하는 태스크 개수
    "static_task_resource_demand_used": False,  # 항상 미리 정해 놓은 태스크 자원 요구량 사용 유무
    "same_task_resource_demand_used": False,  # 각 에피소드 초기에 동일한 태스크 자원 요구량 사용 유무
    "initial_resources_capacity": [70, 80],  # 초기 자원 용량
    "low_demand_resource_at_task": [1, 1],  # 태스크의 각 자원 최소 요구량
    "high_demand_resource_at_task": [20, 20]  # 태스크의 각 자원 최대 요구량
}
if env_config["same_task_resource_demand_used"]:
    assert env_config["static_task_resource_demand_used"] is False

if env_config["static_task_resource_demand_used"]:
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
        self.resource_of_all_tasks_selected = None
        self.cpu_of_all_tasks_selected = None
        self.ram_of_all_tasks_selected = None

        self.min_task_cpu_demand = None
        self.min_task_ram_demand = None

        self.NUM_TASKS = env_config["num_tasks"]
        self.INITIAL_RESOURCES_CAPACITY = env_config["initial_resources_capacity"]
        self.MIN_RESOURCE_DEMAND_AT_TASK = env_config["low_demand_resource_at_task"]
        self.MAX_RESOURCE_DEMAND_AT_TASK = env_config["high_demand_resource_at_task"]
        self.STATIC_TASK_RESOURCE_DEMAND_USED = env_config["static_task_resource_demand_used"]
        self.SAME_TASK_RESOURCE_DEMAND_USED = env_config["same_task_resource_demand_used"]

        self.CPU_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[0]
        self.RAM_RESOURCE_CAPACITY = self.INITIAL_RESOURCES_CAPACITY[1]
        self.SUM_RESOURCE_CAPACITY = sum(self.INITIAL_RESOURCES_CAPACITY)

        self.STATIC_TASK_RESOURCE_DEMAND = [
            [13,  12],
            [14,   9],
            [14,   7],
            [12,  15],
            [13,  14],
            [17,  10],
            [12,  14],
            [ 4,  17],
            [ 8,  14],
            [ 6,  12]
        ]
        
        if self.SAME_TASK_RESOURCE_DEMAND_USED:
            self.TASK_RESOURCE_DEMAND = None

        print("NUM_TASKS:", self.NUM_TASKS)
        print("STATIC_TASK_RESOURCE_DEMAND_USED:", self.STATIC_TASK_RESOURCE_DEMAND_USED)
        print("SAME_TASK_RESOURCE_DEMAND_USED:", self.SAME_TASK_RESOURCE_DEMAND_USED)
        print("min_task_CPU_demand:", self.min_task_cpu_demand)
        print("min_task_RAM_demand:", self.min_task_ram_demand)
        print("###########################################################")

    def get_initial_internal_state(self):
        state = np.zeros(shape=(self.NUM_TASKS + 1, 3), dtype=int)

        if self.STATIC_TASK_RESOURCE_DEMAND_USED:
            state[:-1, 1:] = self.STATIC_TASK_RESOURCE_DEMAND
        else:
            if self.SAME_TASK_RESOURCE_DEMAND_USED:
                if self.TASK_RESOURCE_DEMAND is None:
                    self.TASK_RESOURCE_DEMAND = np.zeros(shape=(self.NUM_TASKS, 2))
                    for task_idx in range(self.NUM_TASKS):
                        self.TASK_RESOURCE_DEMAND[task_idx] = np.random.randint(
                            low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(1, 2)
                        )
                state[:-1, 1:] = self.TASK_RESOURCE_DEMAND
            else:
                for task_idx in range(self.NUM_TASKS):
                    resource_demand = np.random.randint(
                        low=self.MIN_RESOURCE_DEMAND_AT_TASK, high=self.MAX_RESOURCE_DEMAND_AT_TASK, size=(1, 2)
                    )
                    state[task_idx][1:] = resource_demand

        self.min_task_cpu_demand = state[:-1, 1].min()
        self.min_task_ram_demand = state[:-1, 2].min()

        state[-1][1:] = np.array(self.INITIAL_RESOURCES_CAPACITY)

        # print(state)

        return state

    def get_observation_from_internal_state(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.SUM_RESOURCE_CAPACITY
        return observation

    def reset(self, **kwargs):
        self.internal_state = self.get_initial_internal_state()
        self.actions_selected = []
        self.resource_of_all_tasks_selected = 0
        self.cpu_of_all_tasks_selected = 0
        self.ram_of_all_tasks_selected = 0

        observation = self.get_observation_from_internal_state()
        info = None

        return observation, info

    def step(self, action_idx):
        info = {}
        self.actions_selected.append(action_idx)

        step_cpu = self.internal_state[action_idx][1]
        step_ram = self.internal_state[action_idx][2]

        cpu_of_all_tasks_selected_with_this_step = self.cpu_of_all_tasks_selected + step_cpu
        ram_of_all_tasks_selected_with_this_step = self.ram_of_all_tasks_selected + step_ram

        ###########################
        ### terminated 결정 - 시작 ###
        ###########################
        terminated = False
        if self.internal_state[action_idx][0] == 1:
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_FAIL_1   ##### [TYPE 1] The Same Task Selected #####

        elif (cpu_of_all_tasks_selected_with_this_step > self.CPU_RESOURCE_CAPACITY) or \
                    (ram_of_all_tasks_selected_with_this_step > self.RAM_RESOURCE_CAPACITY):
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_FAIL_2   ##### [TYPE 2] Resource Limit Exceeded #####

        else:
            self.internal_state[action_idx][0] = 1
            self.internal_state[action_idx][1] = -1
            self.internal_state[action_idx][2] = -1

            self.cpu_of_all_tasks_selected = cpu_of_all_tasks_selected_with_this_step
            self.ram_of_all_tasks_selected = ram_of_all_tasks_selected_with_this_step

            self.internal_state[-1][1] = self.internal_state[-1][1] - step_cpu
            self.internal_state[-1][2] = self.internal_state[-1][2] - step_ram

            conditions = [
                self.internal_state[-1][1] <= self.min_task_cpu_demand,
                self.internal_state[-1][2] <= self.min_task_ram_demand
            ]

            if 0 not in self.internal_state[:self.NUM_TASKS, 0]:
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_1              ##### All Tasks Allocated Successfully - [ALL] #####
            elif all(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_2              ##### All Resources Used Up - [BEST] #####
            elif any(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_SUCCESS_3              ##### A Resource Used Up - [GOOD] #####
            else:
                pass

        ###########################
        ### terminated 결정 - 종료 ###
        ###########################

        new_observation = self.get_observation_from_internal_state()

        if terminated:
            reward = self.get_reward_information(done_type=info['DoneReasonType'])
        else:
            reward = self.get_reward_information(done_type=None)

        info["CPU_CAPACITY"] = self.CPU_RESOURCE_CAPACITY
        info["RAM_CAPACITY"] = self.RAM_RESOURCE_CAPACITY
        info["ACTIONS_SELECTED"] = self.actions_selected
        info["CPU_RESOURCE_USED"] = self.cpu_of_all_tasks_selected
        info["RAM_RESOURCE_USED"] = self.ram_of_all_tasks_selected
        info["ALL_RESOURCES_USED"] = (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected)
        info["INTERNAL_STATE"] = self.internal_state

        truncated = None

        return new_observation, reward, terminated, truncated, info

    def get_reward_information(self, done_type=None):
        if done_type is None:  # Normal Step
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_FAIL_1:  # The Same Task Selected
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType.TYPE_FAIL_2:  # Resource Limit Exceeded
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType.TYPE_SUCCESS_1:  # All Tasks Allocated Successfully - [ALL]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 2.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_SUCCESS_2:  # Resource allocated fully - [BEST]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 2.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_SUCCESS_3:  # One of Resource allocated fully - [GOOD]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 1.0
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return mission_complete_reward + misbehavior_reward
        # return resource_efficiency_reward + mission_complete_reward + misbehavior_reward
