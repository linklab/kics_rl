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

    def __init__(self, patience, max_num_episodes):
        """
        Args:
            patience (int): episode_reward가 개선될 때까지 기다리는 기간
        """
        self.patience = patience
        self.max_num_episodes = max_num_episodes
        self.counter = 0
        self.max_validation_episode_reward_avg = -np.inf

    def check(
            self, validation_episode_reward_avg, num_tasks, env_name, current_time,
            n_episode, time_steps, training_time_steps, q
    ):
        early_stop = False

        conditions = [
            validation_episode_reward_avg > 0.0,
            validation_episode_reward_avg >= self.max_validation_episode_reward_avg
        ]
        if all(conditions):
            self.model_save(validation_episode_reward_avg, num_tasks, env_name, n_episode, current_time, q)
            self.max_validation_episode_reward_avg = validation_episode_reward_avg
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
                print("[EARLY STOP] COUNTER: {0} - Solved in {1:,} episode, {2:,} steps ({3:,} training steps)!".format(
                    self.counter, n_episode, time_steps, training_time_steps
                ))
            else:
                print("[EARLY STOP] COUNTER: {0}".format(self.counter))
        return early_stop

    def model_save(self, validation_episode_reward_avg, num_tasks, env_name, n_episode, current_time, q):
        filename = "dqn_{0}_{1}_{2}_{3:5.3f}_{4}.pth".format(
            num_tasks, env_name,
            current_time, validation_episode_reward_avg, n_episode
        )
        torch.save(q.state_dict(), os.path.join(MODEL_DIR, filename))
        print("*** MODEL SAVED TO {0}".format(os.path.join(MODEL_DIR, filename)))

        latest_file_name = "dqn_{0}_{1}_latest.pth".format(NUM_TASKS, env_name)
        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, latest_file_name)
        )
        print("*** MODEL UPDATED TO {0}".format(os.path.join(MODEL_DIR, latest_file_name)))


class DQN:
    def __init__(self, env, validation_env, config, env_config, use_wandb):
        self.env = env
        self.validation_env = validation_env
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
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.use_early_stop_with_best_validation_model = config["use_early_stop_with_best_validation_model"]
        self.validation_num_episodes = config["validation_num_episodes"]

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scheduled_percent

        # network
        self.q = QNet(n_features=(NUM_TASKS + 1) * 3, n_actions=NUM_TASKS, device=DEVICE)
        self.target_q = QNet(n_features=(NUM_TASKS + 1) * 3, n_actions=NUM_TASKS, device=DEVICE)

        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size, device=DEVICE)

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

        self.early_stop_model_saver = EarlyStopModelSaver(
            patience=config["early_stop_patience"], max_num_episodes=config["max_num_episodes"]
        )

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

        validation_episode_reward_avg = -1.0

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
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>5.2f},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.3f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:>5,},".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time_str)
                )

            if n_episode % self.train_num_episodes_before_next_validation == 0:
                validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                    validation_episode_reward_lst, validation_episode_reward_avg
                ))

                if self.use_early_stop_with_best_validation_model:
                    is_terminated = self.early_stop_model_saver.check(
                        validation_episode_reward_avg, NUM_TASKS, ENV_NAME, self.current_time,
                        n_episode, self.time_steps, self.training_time_steps, self.q
                    )

            if self.use_wandb:
                self.wandb.log({
                    "[VALIDATE] Mean Episode Reward (Utilization)": validation_episode_reward_avg,
                    "[TRAIN] Episode Reward (Utilization)": episode_reward,
                    "Loss": loss if loss != 0.0 else 0.0,
                    "Epsilon": epsilon,
                    "Episode": n_episode,
                    "Replay buffer": self.replay_buffer.size(),
                    "Training Steps": self.training_time_steps,
                    "Episodes per second": n_episode / total_training_time
                })

            if is_terminated:
                break

        if not self.use_early_stop_with_best_validation_model:
            self.early_stop_model_saver.model_save(
                validation_episode_reward_avg, NUM_TASKS, ENV_NAME, self.max_num_episodes, self.current_time, self.q
            )
        total_training_time = time.time() - total_train_start_time
        total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time_str))
        self.wandb.finish()

    def train(self):
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        observations, actions, next_observations, rewards, dones, action_masks = batch

        q_out = self.q(observations)
        q_values = q_out.gather(dim=1, index=actions)

        with torch.no_grad():
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

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, info = self.validation_env.reset()

            done = False

            while not done:
                action = self.q.get_action(observation, epsilon=0.0, action_mask=info["ACTION_MASK"])

                next_observation, reward, terminated, truncated, info = self.validation_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main():
    env = TaskAllocationEnv(env_config=env_config)
    validation_env = deepcopy(env)

    print("{0:>50}: {1}".format(
        "USE_EARLY_STOP_WITH_BEST_VALIDATION_MODEL", dqn_config["use_early_stop_with_best_validation_model"]
    ))
    print("*" * 100)

    use_wandb = True
    dqn = DQN(
        env=env, validation_env=validation_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
