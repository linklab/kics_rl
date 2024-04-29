# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import time
import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import wandb
from datetime import datetime
from shutil import copyfile
from a_deterministic_actor_and_critic import MODEL_DIR, Actor, Critic, Transition, ReplayBuffer


class DDPG:
    def __init__(self, env, test_env, config, use_wandb):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            self.wandb = wandb.init(
                project="DDPG_{0}".format(self.env_name),
                name=self.current_time,
                config=config
            )

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.policy_freq = config["policy_freq"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_test = config["train_num_episodes_before_next_test"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.exploration_noise = config["exploration_noise"]

        self.actor = Actor(env, n_features=3, n_actions=1)
        self.target_actor = Actor(env, n_features=3, n_actions=1)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.learning_rate)

        self.critic = Critic(n_features=4)
        self.target_critic = Critic(n_features=4)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

    def train_loop(self):
        actor_loss = 0.0
        critic_loss = 0.0
        q_value = 0.0

        total_train_start_time = time.time()

        validation_episode_reward_avg = -1500

        is_terminated = False
        # episode start
        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False
            # timestep start
            while not done:
                self.time_steps += 1
                self.total_time_steps += 1

                if self.total_time_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = self.actor(torch.Tensor(observation))
                        action += torch.normal(0, self.actor.action_scale * self.exploration_noise)
                        action = action.detach().cpu().numpy().clip(self.env.action_space.low,
                                                                    self.env.action_space.high)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.total_time_steps > self.learning_starts:
                    actor_loss, critic_loss, q_value = self.train()

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Actor Loss: {:>7.3f},".format(actor_loss),
                    "Critic Loss: {:>7.3f},".format(critic_loss),
                    "Q-Value: {:>7.3f}".format(q_value),
                    "Training Steps: {:5,}, ".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time)
                )

            if n_episode % self.train_num_episodes_before_next_test == 0:
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

            if self.use_wandb:
                self.wandb.log({
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
                    "[TRAIN] Episode Reward": episode_reward,
                    "[TRAIN] Actor Loss": actor_loss,
                    "[TRAIN] Critic Loss": critic_loss,
                    "[TRAIN] Q-Value": q_value,
                    "[TRAIN] Replay buffer": self.replay_buffer.size(),
                    "Training Episode": n_episode,
                    "Training Steps": self.training_time_steps,
                })

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        self.wandb.finish()

    def train(self):
        actor_loss = 0.0

        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        observations, actions, next_observations, rewards, dones = batch

        # CRITIC UPDATE
        with torch.no_grad():
            next_state_actions = self.target_actor(next_observations)
            next_target_critic = self.target_critic(next_observations, next_state_actions)
            next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (next_target_critic).view(-1)

        current_q_value = self.critic(observations, actions).view(-1)
        critic_loss = F.mse_loss(current_q_value, next_q_value)

        # optimize the model
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ACTOR UPDATE
        if self.total_time_steps % self.policy_freq == 0:
            actor_loss = -self.critic(observations, self.actor(observations)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update the target network
            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # print(
        #     q_values.shape, values.shape, advantages.shape, values.shape, action_log_probs.shape, log_pi_advantages.shape,
        #     entropy.shape, entropy_sum.shape, log_pi_advantages_sum.shape, "!!!"
        # )

        return (
            actor_loss.item(),
            critic_loss.item(),
            current_q_value.mean().item()
        )

    def model_save(self, validation_episode_reward_avg):
        filename = "ddpg_{0}_{1:4.1f}_{2}.pth".format(
            self.env_name, validation_episode_reward_avg, self.current_time
        )
        torch.save(self.actor.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, "ddpg_{0}_latest.pth".format(self.env_name))
        )

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                # action = self.actor.get_action(observation)
                action = self.actor(torch.Tensor(observation)).detach().numpy()

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main():
    ENV_NAME = "Pendulum-v1"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,                       # 환경의 이름
        "max_num_episodes": 200_000,                # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                          # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0003,                    # 학습율
        "replay_buffer_size": 200_000,              # 리플레이 버퍼 사이즈
        "gamma": 0.98,                              # 감가율
        "tau": 0.005,                               # soft_target_update를 위한 가중치 변수
        "print_episode_interval": 20,               # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_test": 100, # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,               # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -150,          # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "learning_starts": 12_000,                  # exploration과 learning의 기준 step
        "policy_freq": 1,                           # policy 훈련 주기
        "exploration_noise": 0.1,                   # 노이즈
    }

    use_wandb = True
    ddpg = DDPG(
        env=env, test_env=test_env, config=config, use_wandb=use_wandb
    )
    ddpg.train_loop()


if __name__ == '__main__':
    main()
