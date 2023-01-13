# Problem: Multiple Tasks Allocation to One Computing server
import gymnasium as gym
import numpy as np
import copy
import enum


class DoneReasonType(enum.Enum):
    TYPE_1 = "The Same Task Selected"
    TYPE_2 = "Resource Limit Exceeded"
    TYPE_3 = "All Resources allocated fully - [BEST]"
    TYPE_4 = "A Resource allocated fully - [GOOD]"
    TYPE_5 = "All Tasks Selected - [ALL]"


class TaskAllocationEnv(gym.Env):
    def __init__(
            self,
            INITIAL_TASK_DISTRIBUTION_FIXED,
            NUM_TASKS,
            LOW_DEMAND_RESOURCE_AT_TASK,
            HIGH_DEMAND_RESOURCE_AT_TASK,
            INITIAL_RESOURCES_CAPACITY
    ):
        super(TaskAllocationEnv, self).__init__()
        self.internal_state = None
        self.actions_selected = None
        self.resource_of_all_tasks_selected = None
        self.cpu_of_all_tasks_selected = None
        self.ram_of_all_tasks_selected = None

        self.fixed_initial_internal_state = None

        self.min_task_cpu_demand = None
        self.min_task_ram_demand = None

        self.INITIAL_TASK_DISTRIBUTION_FIXED = INITIAL_TASK_DISTRIBUTION_FIXED
        self.NUM_TASKS = NUM_TASKS
        self.LOW_DEMAND_RESOURCE_AT_TASK = LOW_DEMAND_RESOURCE_AT_TASK
        self.HIGH_DEMAND_RESOURCE_AT_TASK = HIGH_DEMAND_RESOURCE_AT_TASK
        self.INITIAL_RESOURCES_CAPACITY = INITIAL_RESOURCES_CAPACITY
        self.SUM_RESOURCE_CAPACITY = sum(self.INITIAL_RESOURCES_CAPACITY)
        self.CPU_RESOURCE_CAPACITY = INITIAL_RESOURCES_CAPACITY[0]
        self.RAM_RESOURCE_CAPACITY = INITIAL_RESOURCES_CAPACITY[1]

        self.STATIC_RESOURCE_DEMAND = [
            [13,  12],
            [14,   9],
            [14,   7],
            [12,  15],
            [13,  14],
            [17,  10],
            [12,  14],
            [4,  17],
            [8,  14],
            [6,  12]
        ]

        if self.INITIAL_TASK_DISTRIBUTION_FIXED:
            self.fixed_initial_internal_state = self.get_initial_internal_state()

            print("Initial Internal State\n", self.fixed_initial_internal_state)
            print("min_task_CPU_demand:", self.min_task_cpu_demand)
            print("min_task_RAM_demand:", self.min_task_ram_demand)
            print("###########################################################")

    def get_initial_internal_state(self):
        state = np.zeros(shape=(self.NUM_TASKS + 1, 3), dtype=int)

        for task_idx in range(self.NUM_TASKS):
            state[task_idx][1:] = self.STATIC_RESOURCE_DEMAND[task_idx]

        # for task_idx in range(self.NUM_TASKS):
        #     resource_demand = np.random.randint(
        #         low=self.LOW_DEMAND_RESOURCE_AT_TASK, high=self.HIGH_DEMAND_RESOURCE_AT_TASK, size=(1, 2)
        #     )
        #     state[task_idx][1:] = resource_demand

        self.min_task_cpu_demand = state[:-1, 1].min()
        self.min_task_ram_demand = state[:-1, 2].min()

        state[-1][1:] = np.array(self.INITIAL_RESOURCES_CAPACITY)

        return state

    def get_observation_from_internal_state(self):
        observation = copy.deepcopy(self.internal_state.flatten()) / self.SUM_RESOURCE_CAPACITY
        return observation

    def reset(self, **kwargs):
        if self.INITIAL_TASK_DISTRIBUTION_FIXED:
            assert self.fixed_initial_internal_state is not None
            self.internal_state = copy.deepcopy(self.fixed_initial_internal_state)
        else:
            self.internal_state = self.get_initial_internal_state()

        #print(self.internal_state, "!!!!!!!!!!!!!!!!!!")

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

        if self.internal_state[action_idx][0] == 1:
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_1   ##### [TYPE 1] The Same Task Selected #####

        elif (cpu_of_all_tasks_selected_with_this_step > self.CPU_RESOURCE_CAPACITY) or \
                    (ram_of_all_tasks_selected_with_this_step > self.RAM_RESOURCE_CAPACITY):
            terminated = True
            info['DoneReasonType'] = DoneReasonType.TYPE_2   ##### [TYPE 2] Resource Limit Exceeded #####

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

            if all(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_3              ##### [TYPE 3] All Resources allocated fully - [BEST] #####

            elif any(conditions):
                terminated = True
                info['DoneReasonType'] = DoneReasonType.TYPE_4              ##### [TYPE 4] A Resource allocated fully - [GOOD] #####

            else:
                if 0 not in self.internal_state[:self.NUM_TASKS, 0]:
                    terminated = True
                    info['DoneReasonType'] = DoneReasonType.TYPE_5  ##### [TYPE 5] All Tasks Selected - [ALL] #####

                else:
                    terminated = False  ##### It's Normal Step (Not terminated)

        new_observation = self.get_observation_from_internal_state()

        if terminated:
            reward = self.get_reward_information(done_type=info['DoneReasonType'])
        else:
            reward = self.get_reward_information(done_type=None)

        info["ACTIONS_SELECTED"] = self.actions_selected
        info["RESOURCE_ALLOCATED"] = (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected)
        info["CPU_RESOURCE_ALLOCATED"] = self.cpu_of_all_tasks_selected
        info["RAM_RESOURCE_ALLOCATED"] = self.ram_of_all_tasks_selected
        info["CPU_LIMIT"] = self.CPU_RESOURCE_CAPACITY
        info["RAM_LIMIT"] = self.RAM_RESOURCE_CAPACITY
        info["INTERNAL_STATE"] = self.internal_state

        truncated = None

        return new_observation, reward, terminated, truncated, info

    def get_reward_information(self, done_type=None):
        if done_type is None:  # Normal Step
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_1:  # The Same Task Selected
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType.TYPE_2:  # Resource Limit Exceeded
            resource_efficiency_reward = 0.0
            mission_complete_reward = 0.0
            misbehavior_reward = -1.0

        elif done_type == DoneReasonType.TYPE_3:  # Resource allocated fully - [BEST]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 2.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_4:  # One of Resource allocated fully - [GOOD]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 1.0
            misbehavior_reward = 0.0

        elif done_type == DoneReasonType.TYPE_5:  # All Tasks Selected - [ALL]
            resource_efficiency_reward = np.tanh(
                (self.cpu_of_all_tasks_selected + self.ram_of_all_tasks_selected) / self.SUM_RESOURCE_CAPACITY
            )
            mission_complete_reward = 2.0
            misbehavior_reward = 0.0

        else:
            raise ValueError()

        return resource_efficiency_reward + mission_complete_reward + misbehavior_reward
