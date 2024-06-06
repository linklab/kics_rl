# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import os
import random
import time
from dataclasses import dataclass
from shutil import copyfile

import gymnasium as gym
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from q_net import QNetwork, MODEL_DIR
from buffers import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "DDDQN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 1000000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 50000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""
    print_episode_interval: int = 10
    """Episode 통계 출력에 관한 에피소드 간격"""
    train_num_episodes_before_next_test: int = 50
    """validation 사이마다 각 훈련 episode 간격"""
    episode_reward_avg_solved: int = 400
    """훈련 조기 종료 validation reward cut"""
    save_every_n_episodes: int = 1000
    """모델 세이브 에피소드 주기"""
    validation_num_episodes: int = 3
    """validation에 수행하는 episode 횟수"""


args = tyro.cli(Args)


class DDDQN:
    def __init__(self, env, test_env, run_name, use_wandb):
        self.envs = env
        self.test_env = test_env
        self.run_name = run_name
        self.use_wandb = use_wandb

        self.env_name = args.env_id

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            self.wandb = wandb.init(
                project="DDDQN_{0}".format(self.env_name),
                name=self.current_time,
                config=vars(args)
            )

        # network
        self.q_network = QNetwork(self.envs).to(DEVICE)
        self.target_network = QNetwork(self.envs).to(DEVICE)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=args.learning_rate)

        # agent
        self.rb = ReplayBuffer(
            args.buffer_size,
            self.envs.single_observation_space,
            self.envs.single_action_space,
            DEVICE,
            optimize_memory_usage=True,
            handle_timeout_termination=False,
        )

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

    def train_loop(self):
        # episode counter
        n_episode = 1
        episode_reward = 0
        loss = 0.0

        validation_episode_reward_avg = 0.0

        total_train_start_time = time.time()

        is_terminated = False

        # TRY NOT TO MODIFY: start the game
        obs, _ = self.envs.reset()
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps,
                                       global_step)
            actions = self.q_network.get_action(torch.Tensor(obs).to(DEVICE), epsilon)

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

            episode_reward += rewards.squeeze(-1)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
            self.rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    loss = self.train()
                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(self.target_network.parameters(),
                                                                     self.q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                        )

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if n_episode % args.print_episode_interval == 0:
                            total_training_time = time.time() - total_train_start_time
                            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
                            print(
                                "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, global_step),
                                "Episode Reward: {:>5},".format(episode_reward),
                                "Replay buffer: {:>6,},".format(self.rb.size()),
                                "Loss: {:6.5f},".format(loss),
                                "Epsilon: {:4.2f},".format(epsilon),
                                "Elapsed Time: {}".format(total_training_time)
                            )
                        if n_episode % args.train_num_episodes_before_next_test == 0:
                            validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                            print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                                validation_episode_reward_lst, validation_episode_reward_avg
                            ))

                            if n_episode % args.save_every_n_episodes == 0:
                                self.save_model(validation_episode_reward_avg)

                            if validation_episode_reward_avg > args.episode_reward_avg_solved:
                                print("Solved in {0:,} steps ({1:,} training steps)!".format(
                                    self.time_steps, self.training_time_steps
                                ))
                                self.save_model(validation_episode_reward_avg)
                                is_terminated = True

                        n_episode += 1
                        episode_reward = 0

            if self.use_wandb:
                self.wandb.log({
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(args.validation_num_episodes): validation_episode_reward_avg,
                    "[TRAIN] Episode Reward": episode_reward,
                    "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                    "[TRAIN] Epsilon": epsilon,
                    "[TRAIN] Replay buffer": self.rb.size(),
                    "Training Episode": n_episode,
                    "Training Steps": self.training_time_steps
                })

            if is_terminated:
                break
        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        self.wandb.finish()

    def train(self):
        data = self.rb.sample(args.batch_size)
        q_out = self.q_network(data.observations)
        q_values = q_out.gather(dim=1, index=data.actions)

        with torch.no_grad():
            q_prime_out = self.q_network(data.next_observations)  # online network를 사용하여 다음 상태에서의 행동을 선택
            best_action = q_prime_out.argmax(dim=1)
            target_q_values = self.target_network(data.next_observations)
            target_q_values[data.dones.squeeze(-1)] = 0.0

            # Calculate the targets
            targets = data.rewards + args.gamma * target_q_values.gather(dim=1, index=best_action.unsqueeze(-1))

        # loss is just scalar torch value
        loss = F.mse_loss(targets.detach(), q_values)

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_model(self, validation_episode_reward_avg):
        filename = "dddqn_{0}_{1:4.1f}_{2}.pth".format(
            self.env_name, validation_episode_reward_avg, self.current_time
        )
        torch.save(self.q_network.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, "double_dueling_dqn_{0}_latest.pth".format(self.env_name))
        )

    def validate(self):
        episode_reward_lst = np.zeros(shape=(args.validation_num_episodes,), dtype=float)

        for i in range(args.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.q_network.get_action(torch.Tensor(observation).to(DEVICE), epsilon=0.0)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward.squeeze(-1)
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))  # (210, 160, 3) >> (84, 84, 3)
        env = gym.wrappers.GrayScaleObservation(env)    # (84, 84, 3) >> (84, 84, 1)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def main():
    run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"

    env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    test_env = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, 0, args.capture_video, run_name)]
    )

    use_wandb = True
    dddqn = DDDQN(
        env=env, test_env=test_env, run_name=run_name, use_wandb=use_wandb
    )
    dddqn.train_loop()


if __name__ == "__main__":
    main()
