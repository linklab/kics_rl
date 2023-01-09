# https://gymnasium.farama.org/environments/classic_control/cart_pole/
# -*- coding: utf-8 -*-
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import wandb
from datetime import datetime

from c_policy import DEVICE, MODEL_DIR, Policy, Transition, Buffer, StateValueNet


class REINFORCE:
    def __init__(self, env, validation_env, config, use_baseline, use_wandb):
        self.env = env
        self.validation_env = validation_env
        self.use_baseline = use_baseline
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            self.wandb = wandb.init(
                project="DQN_{0}".format(self.env_name),
                name=self.current_time,
                config=config
            )

        self.max_num_episodes = config["max_num_episodes"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.validation_episode_interval = config["validation_episode_interval"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

        self.policy = Policy(n_features=3, n_actions=1)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        if self.use_baseline:
            self.state_value_net = StateValueNet(n_features=3)
            self.value_optimizer = optim.Adam(self.state_value_net.parameters(), lr=self.learning_rate)

        self.buffer = Buffer()

        self.time_steps = 0
        self.training_time_steps = 0

    def train_loop(self):
        total_train_start_time = time.time()

        validation_episode_reward_avg = -2000

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

            # TRAIN AFTER EPISODE DONE
            objective = self.train()
            self.buffer.clear()

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Objective: {:>7.3f},".format(objective),
                    "Training Steps: {:5,}".format(self.training_time_steps),
                    "Total Elapsed Time: {}".format(total_training_time)
                )

            if self.training_time_steps > 0 and n_episode % self.validation_episode_interval == 0:
                validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                    validation_episode_reward_lst, validation_episode_reward_avg
                ))

                if validation_episode_reward_avg > self.episode_reward_avg_solved:
                    print("Solved in {0:,} steps ({1:,} training steps)!".format(
                        self.time_steps, self.training_time_steps
                    ))
                    self.model_save(validation_episode_reward_avg)
                    is_terminated = True

            if self.use_wandb and n_episode % 2 == 0:
                self.wandb.log({
                    "[VALIDATE] Mean Episode Reward": validation_episode_reward_avg,
                    "[TRAIN] Episode Reward": episode_reward,
                    "Objective": objective if objective != 0.0 else 0.0,
                    "Episode": n_episode,
                    "Training Steps": self.training_time_steps,
                })

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

    def train(self):
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.buffer.get()

        # rewards = torch.flip(rewards, dims=[0])
        rewards = rewards.squeeze(dim=1).cpu().numpy()[::-1]

        G = 0.0
        return_lst = []
        for reward in rewards:
            G = reward + self.gamma * G
            return_lst.append(G)

        returns = torch.tensor(return_lst[::-1], dtype=torch.float32, device=DEVICE)

        mu_v, std_v = self.policy.forward(observations)
        dist = Normal(loc=mu_v, scale=std_v)
        action_log_probs = dist.log_prob(value=actions).squeeze(dim=1)

        if self.use_baseline:
            values = self.state_value_net(observations).squeeze(dim=1)
            v_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()

            returns_baseline = returns - values.detach()
            returns_baseline = (returns_baseline - torch.mean(returns_baseline)) / (torch.std(returns_baseline) + 1e-7)
            log_pi_returns = action_log_probs * returns_baseline
            policy_objective = torch.sum(log_pi_returns)
            #print(returns.shape, values.shape, returns_baseline.shape, action_log_probs.shape, log_pi_returns.shape, policy_objective.shape, "!!!")
        else:
            returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-7)
            log_pi_returns = action_log_probs * returns
            policy_objective = torch.sum(log_pi_returns)
            # print(returns.shape, action_log_probs.shape, log_pi_returns.shape, policy_objective.shape, "!!!")

        loss = -1.0 * policy_objective

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return policy_objective

    def model_save(self, validation_episode_reward_avg):
        torch.save(
            self.policy.state_dict(),
            os.path.join(MODEL_DIR, "vpg_{0}_{1:4.1f}.pth".format(self.env_name, validation_episode_reward_avg))
        )

    def validate(self):
        episode_reward_lst = []

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.validation_env.reset()

            done = False

            while not done:
                action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.validation_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst.append(episode_reward)

        return episode_reward_lst, np.average(episode_reward_lst)


def main():
    ENV_NAME = "Pendulum-v1"

    # env
    env = gym.make(ENV_NAME)
    validation_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,                       # 환경의 이름
        "max_num_episodes": 100_000,                  # 훈련을 위한 최대 에피소드 횟수
        "learning_rate": 0.00005,                    # 학습율
        "gamma": 0.99,                              # 감가율
        "print_episode_interval": 20,               # Episode 통계 출력에 관한 에피소드 간격
        "validation_episode_interval": 100,         # 검증을 위한 episode 간격
        "validation_num_episodes": 3,               # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -10,           # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }

    reinforce = REINFORCE(env=env, validation_env=validation_env, config=config, use_baseline=True, use_wandb=True)
    reinforce.train_loop()


if __name__ == '__main__':
    main()
