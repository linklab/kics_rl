# https://gymnasium.farama.org/environments/classic_control/pendulum/
import time
import os
import multiprocessing
import copy

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

from c_actor_and_critic import MODEL_DIR, Actor, Critic, Transition, Buffer


class A3CAgent:
    def __init__(self,
                 worker_id,
                 global_actor,
                 global_critic,
                 global_actor_optimizer,
                 global_critic_optimizer,
                 global_counter,
                 use_wandb,
                 env,
                 test_env,
                 config):
        self.env_name = config["env_name"]
        self.env = env
        self.test_env = test_env

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.local_actor = copy.deepcopy(global_actor)
        self.local_actor.load_state_dict(global_actor.state_dict())
        self.local_critic = copy.deepcopy(global_critic)
        self.local_critic.load_state_dict(global_critic.state_dict())

        self.actor_optimizer = global_actor_optimizer
        self.critic_optimizer = global_critic_optimizer

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.entropy_beta = config["entropy_beta"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episode_before_next_test = config["train_num_episodes_before_next_test"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

        self.buffer = Buffer()

        self.time_steps = 0
        self.training_time_steps = 0

        self.use_wandb = use_wandb

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            self.wandb = wandb.init(
                project="A3C_{0}".format(self.env_name),
                name=self.current_time,
                config=config
            )

    def train_loop(self, global_counter):
        total_train_start_time = time.time()

        policy_loss = critic_loss = avg_mu_v = avg_std_v = avg_action = avg_action_prob = 0.0

        is_terminated = False

        # Additional attributes
        last_validation_step = 0

        for n_episode in range(1, self.max_num_episodes + 1):
            # print(global_counter.value, "global_counter")

            self.local_actor.load_state_dict(self.global_actor.state_dict())
            self.local_critic.load_state_dict(self.global_critic.state_dict())

            global_counter.value += 1

            episode_reward = 0

            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1
                action = self.local_actor.get_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action * 2)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.batch_size == 0:
                    policy_loss, critic_loss, avg_mu_v, avg_std_v, avg_action, avg_action_prob = self.train()

                    self.buffer.clear()

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Police Loss: {:>7.3f},".format(policy_loss),
                    "Critic Loss: {:>7.3f},".format(critic_loss),
                    "Training Steps: {:5,},".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time)
                )

            if 0 == n_episode % self.train_num_episode_before_next_test:
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
                    # break

            if not self.use_wandb:
                pass
            else:
                self.wandb.log({
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                        self.validation_num_episodes): episode_reward,
                    "[TRAIN] Episode Reward": episode_reward,
                    "[TRAIN] Policy Loss": policy_loss,
                    "[TRAIN] Critic Loss": critic_loss,
                    "[TRAIN] avg_mu_v": avg_mu_v,
                    "[TRAIN] avg_std_v": avg_std_v,
                    "[TRAIN] avg_action": avg_action,
                    "[TRAIN] avg_action_prob": avg_action_prob,
                    "Training Episode": n_episode,
                    "Training Steps": self.training_time_steps,
                })

            with global_counter.get_lock():
                if global_counter.value - last_validation_step >= 100:
                    last_validation_step = global_counter.value
                    validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                    print("[Validation] Episode Reward List: {0}, Average: {1:.3f}".format(
                        validation_episode_reward_lst, validation_episode_reward_avg
                    ))

            if is_terminated:
                break

    def train(self):
        self.training_time_steps += 1

        # Getting values from buffer
        observations, actions, next_observations, rewards, dones = self.buffer.get()

        # Calculating target values
        values = self.local_critic(observations).squeeze(dim=-1)
        next_values = self.local_critic(next_observations).squeeze(dim=-1)
        next_values[dones] = 0.0

        q_values = rewards.squeeze(
            dim=-1) + self.gamma * next_values  # Отделенный для предотвращения распространения градиентов в нижний критик через цели

        # CRITIC UPDATE
        critic_loss = F.mse_loss(q_values.detach(), values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Advantage calculating
        advantages = q_values - values
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

        # Actor Loss computing
        mu, std = self.local_actor.forward(observations)
        dist = Normal(mu, std)
        action_log_probs = dist.log_prob(value=actions).squeeze(dim=-1)  # natural log

        log_pi_advantages = action_log_probs * advantages.detach()
        log_pi_advantages_sum = log_pi_advantages.sum()

        entropy = dist.entropy().squeeze(dim=-1)
        entropy_sum = entropy.sum()

        actor_loss = -1.0 * log_pi_advantages_sum - 1.0 * entropy_sum * self.entropy_beta

        # Actor Update
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return (
            actor_loss.item(), critic_loss.item(), mu.mean().item(), std.mean().item(),
            actions.type(torch.float32).mean().item(), action_log_probs.exp().mean().item()
        )

    def validate(self):
        episode_rewards = np.zeros(self.validation_num_episodes)

        # test_env = gym.make(self.env_name)

        # Synchronize the global model to the local model
        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

        for i in range(self.validation_num_episodes):
            # test_env = gym.make(self.env_name)
            observation, _ = self.test_env.reset()
            episode_reward = 0

            done = False

            while not done:
                action = self.global_actor.get_action(observation, exploration=False)
                next_observation, reward, terminated, truncated, _ = self.test_env.step(action * 2)
                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_rewards[i] = episode_reward

        return episode_rewards, np.average(episode_rewards)

    def model_save(self, validation_episode_reward_avg):
        filename = f"a3c_{self.env_name}_{validation_episode_reward_avg:.1f}_{self.current_time}.pth"
        torch.save({
            'actor_state_dict': self.local_actor.actor.state_dict(),
            'critic_state_dict': self.local_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'validation_reward': validation_episode_reward_avg
        }, os.path.join(MODEL_DIR, filename))
        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, f"a3c_{self.env_name}_latest.pth")
        )


def worker(process_id, global_actor, global_critic, actor_optimizer, critic_optimizer, global_counter, use_wandb, test_env,
           config):
    env_name = config["env_name"]
    env = gym.make(env_name)

    agent = A3CAgent(
        worker_id=process_id,
        global_actor=global_actor,
        global_critic=global_critic,
        global_actor_optimizer=actor_optimizer,
        global_critic_optimizer=critic_optimizer,
        global_counter=global_counter,
        use_wandb=use_wandb,
        env=env,
        test_env=test_env,
        config=config)

    agent.train_loop(global_counter)




def main():
    ENV_NAME = "Pendulum-v1"

    config = {
        "env_name": ENV_NAME,  # 환경의 이름
        "num_workers": 4,
        "max_num_episodes": 50_000,  # 훈련을 위한 최대 에피소드 횟수
        # "batch_size": 64,
        "batch_size": 32,  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0005,  # 학습율
        "gamma": 0.99,  # 감가율
        "entropy_beta": 0.01,  # 엔트로피 가중치
        "print_episode_interval": 20,  # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_test": 100,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,  # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -200  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }

    use_wandb = True
    num_workers = min(config["num_workers"], multiprocessing.cpu_count() - 1)

    test_env = gym.make(ENV_NAME)

    # Initialize global models and optimizers
    global_actor = Actor(n_features=3, n_actions=1).share_memory()
    global_critic = Critic(n_features=3).share_memory()

    actor_optimizer = optim.Adam(global_actor.parameters(), lr=config["learning_rate"])
    critic_optimizer = optim.Adam(global_critic.parameters(), lr=config["learning_rate"])




    # Create global_counter variable in multiprocessing shared memory
    global_counter = multiprocessing.Value('i', 0)

    processes = []

    for i in range(num_workers):
        p = multiprocessing.Process(
            target=worker,
            args=(i, global_actor, global_critic, actor_optimizer, critic_optimizer, global_counter, use_wandb, test_env, config)
        )
        p.start()
        processes.append(p)

    while True:
        time.sleep(0.1)
        print("global_counter", global_counter.value)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
