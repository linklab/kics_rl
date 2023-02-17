# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import time
import os
from copy import deepcopy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime
from shutil import copyfile

from a_config import env_config, dqn_config, ENV_NAME, NUM_TASKS
from c_task_allocation_env import TaskAllocationEnv
from e_qnet import QNet, ReplayBuffer, Transition, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopModelSaver:
    """주어진 patience 이후로 episode_reward가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, patience):
        """
        Args:
            patience (int): episode_reward가 개선될 때까지 기다리는 기간
        """
        self.patience = patience
        self.counter = 0
        self.max_test_episode_reward = -np.inf
        self.model_filename_saved = None

    def check(
            self, test_episode_reward_avg, num_tasks, env_name, current_time,
            n_episode, time_steps, training_time_steps, q
    ):
        early_stop = False

        if test_episode_reward_avg >= self.max_test_episode_reward:
            print("[EARLY STOP] test_episode_reward {0:.5f} is increased to {1:.5f}".format(
                self.max_test_episode_reward, test_episode_reward_avg
            ))
            self.model_save(test_episode_reward_avg, num_tasks, env_name, n_episode, current_time, q)
            self.max_test_episode_reward = test_episode_reward_avg
            self.counter = 0
        else:
            self.counter += 1
            if self.counter < self.patience:
                print("[EARLY STOP] COUNTER: {0} (test_episode_reward/max_test_episode_reward={1:.5f}/{2:.5f})".format(
                    self.counter, test_episode_reward_avg, self.max_test_episode_reward
                ))
            else:
                early_stop = True
                print("[EARLY STOP] COUNTER: {0} - Solved in {1:,} episode, {2:,} steps ({3:,} training steps)!".format(
                    self.counter, n_episode, time_steps, training_time_steps
                ))
        return early_stop

    def model_save(self, test_episode_reward_avg, num_tasks, env_name, n_episode, current_time, q):
        if self.model_filename_saved is not None:
            os.remove(self.model_filename_saved)

        filename = "dqn_{0}_{1}_{2}_{3:5.3f}_{4}.pth".format(
            num_tasks, env_name,
            current_time, test_episode_reward_avg, n_episode
        )
        filename = os.path.join(MODEL_DIR, filename)
        torch.save(q.state_dict(), filename)
        self.model_filename_saved = filename
        print("*** MODEL SAVED TO {0}".format(filename))

        latest_file_name = "dqn_{0}_{1}_latest.pth".format(NUM_TASKS, env_name)
        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, latest_file_name)
        )
        print("*** MODEL UPDATED TO {0}".format(os.path.join(MODEL_DIR, latest_file_name)))


class DQN:
    def __init__(self, env, test_env, config, env_config, use_wandb):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = ENV_NAME

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            project_name = "DQN_{0}_{1}_WITH_{2}".format(
                NUM_TASKS, self.env_name, config["max_num_episodes"]
            )
            if env_config["use_static_task_resource_demand"] is False and \
                    env_config["use_same_task_resource_demand"] is False:
                project_name += "_RANDOM_INSTANCES"
            elif env_config["use_static_task_resource_demand"] is True:
                project_name += "_STATIC_INSTANCES"
            elif env_config["use_same_task_resource_demand"] is True:
                project_name += "_SAME_INSTANCES"
            else:
                raise ValueError()

            self.wandb = wandb.init(
                project=project_name,
                name="{0}_{1}".format("AM", self.current_time),
                config=config | env_config
            )

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.steps_between_train = config["steps_between_train"]
        self.target_sync_step_interval = config["target_sync_step_interval"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_final_scheduled_percent = config["epsilon_final_scheduled_percent"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_test = config["train_num_episodes_before_next_test"]
        self.test_num_episodes = config["test_num_episodes"]
        self.double_dqn = config["double_dqn"]

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scheduled_percent

        # network
        self.q = QNet(n_features=(NUM_TASKS + 1) * 4, n_actions=NUM_TASKS, device=DEVICE)
        self.target_q = QNet(n_features=(NUM_TASKS + 1) * 4, n_actions=NUM_TASKS, device=DEVICE)

        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, device=DEVICE)

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

        self.early_stop_model_saver = EarlyStopModelSaver(patience=config["early_stop_patience"])

    def epsilon_scheduled(self, current_episode):
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)

        epsilon = min(
            self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start),
            self.epsilon_start
        )
        return epsilon

    def train_loop(self):
        loss = 0.0

        total_train_start_time = time.time()
        test_episode_reward_avg = 0.0
        test_total_value_avg = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0
            observation, info = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1
                self.total_time_steps += 1

                action = self.q.get_action(observation, epsilon, action_mask=info["ACTION_MASK"])

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated, info["ACTION_MASK"])

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.total_time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    loss = self.train()

            total_training_time = time.time() - total_train_start_time
            total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {0:4,}/{1:5,}, Time Steps {2:6,}]".format(
                        n_episode, self.max_num_episodes, self.time_steps
                    ),
                    "Episode Reward: {:>4.2f},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:.6f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:>5,},".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time_str)
                )

            # print(epsilon, self.epsilon_end, n_episode, self.train_num_episodes_before_next_test, "!!!")
            if epsilon <= self.epsilon_end + 1e-6 and n_episode % self.train_num_episodes_before_next_test == 0:
                test_episode_reward_lst, test_episode_reward_avg, test_total_value_lst, test_total_value_avg = \
                    self.test()

                print("[Test Episode Reward: {0}] Average: {1:.3f}".format(
                    test_episode_reward_lst, test_episode_reward_avg
                ))
                print("[Test Total Value: {0}] Average: {1:.3f}".format(
                    test_total_value_lst, test_total_value_avg
                ))

                is_terminated = self.early_stop_model_saver.check(
                    test_episode_reward_avg=test_episode_reward_avg,
                    num_tasks=NUM_TASKS, env_name=ENV_NAME, current_time=self.current_time,
                    n_episode=n_episode, time_steps=self.time_steps, training_time_steps=self.training_time_steps,
                    q=self.q
                )

            if self.use_wandb:
                self.wandb.log({
                    "[TEST] Mean Episode Reward ({0} Episodes)".format(self.test_num_episodes): test_episode_reward_avg,
                    "[TEST] Mean Total Value ({0} Episodes)".format(self.test_num_episodes): test_total_value_avg,
                    "[TRAIN] Episode Reward (Total Value)": episode_reward,
                    "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                    "[TRAIN] Epsilon": epsilon,
                    "[TRAIN] Replay buffer": self.replay_buffer.size(),
                    "[TRAIN] Episodes per second": n_episode / total_training_time,
                    "Training Episode": n_episode,
                    "Training Steps": self.training_time_steps
                })

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time_str))
        self.wandb.finish()

    def train(self):
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        observations, actions, next_observations, rewards, dones, action_masks = batch

        q_out = self.q(observations)
        q_values = q_out.gather(dim=-1, index=actions)

        with torch.no_grad():
            if self.double_dqn:
                q_prime_out = self.q(next_observations)
                q_prime_out = q_prime_out.masked_fill(action_masks, -float('inf'))
                # target_argmax_action.shape: [256, 1]
                target_argmax_action = torch.argmax(q_prime_out, dim=-1, keepdim=True)

                q_prime_out = self.target_q(next_observations)
                next_q_values = q_prime_out.gather(dim=-1, index=target_argmax_action)
                next_q_values[dones] = 0.0

                targets = rewards + self.gamma * next_q_values
            else:
                q_prime_out = self.target_q(next_observations)

                q_prime_out = q_prime_out.masked_fill(action_masks, -float('inf'))

                max_q_prime = q_prime_out.max(dim=-1, keepdim=True).values
                max_q_prime[dones] = 0.0

                targets = rewards + self.gamma * max_q_prime

        loss = F.mse_loss(targets.detach(), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # sync
        if self.time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

    def test(self):
        episode_reward_lst = np.zeros(shape=(self.test_num_episodes,), dtype=float)
        total_value_lst = np.zeros(shape=(self.test_num_episodes,), dtype=float)

        for i in range(self.test_num_episodes):
            episode_reward = 0

            observation, info = self.test_env.reset()

            done = False

            while not done:
                action = self.q.get_action(observation, epsilon=0.0, action_mask=info["ACTION_MASK"])

                next_observation, reward, terminated, truncated, info = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward
            total_value_lst[i] = info["VALUE_ALLOCATED"]

        return episode_reward_lst, np.average(episode_reward_lst), total_value_lst, np.average(total_value_lst)


def main():
    env = TaskAllocationEnv(env_config=env_config)
    test_env = deepcopy(env)

    print("*" * 100)

    use_wandb = True
    dqn = DQN(
        env=env, test_env=test_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
