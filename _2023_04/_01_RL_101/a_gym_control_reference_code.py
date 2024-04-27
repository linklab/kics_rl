# Create environment
import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")
import numpy as np
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(edgeitems=10, linewidth=100000, formatter=dict(float=lambda x: "%4.2f" % x))

## BOX2D
# https://gymnasium.farama.org/environments/box2d/
#env = gym.make("LunarLander-v2", render_mode="human")

## ATARI
# https://gymnasium.farama.org/environments/atari/
#env = gym.make("ALE/Breakout-v5", render_mode="human")

## MUJOCO
# https://gymnasium.farama.org/environments/mujoco/
# env = gym.make("Ant-v4",                        render_mode="human")
# env = gym.make("HalfCheetah-v4",                render_mode="human")
# env = gym.make("Hopper-v4",                     render_mode="human")
# env = gym.make("HumanoidStandup-v4",            render_mode="human")
# env = gym.make("Humanoid-v4",                   render_mode="human")
# env = gym.make("InvertedDoublePendulum-v4",     render_mode="human")
# env = gym.make("InvertedPendulum-v4",           render_mode="human")
# env = gym.make("Pusher-v4",                     render_mode="human")
# env = gym.make("Reacher-v4",                    render_mode="human")
# env = gym.make("Swimmer-v4",                    render_mode="human")
# env = gym.make("Walker2d-v4",                   render_mode="human")

# gymnasium-robotics
# pip install gymnasium-robotics[mujoco-py]

# FETCH
# env = gym.make('FetchSlide-v2', max_episode_steps=100, render_mode="human")
# env = gym.make('FetchPickAndPlace-v2', max_episode_steps=100, render_mode="human")
# env = gym.make('FetchReach-v2', max_episode_steps=100, render_mode="human")
# env = gym.make('FetchPush-v2', max_episode_steps=100, render_mode="human")

# Shadow Dexterous Hand
# env = gym.make('HandReach-v1', max_episode_steps=100, render_mode="human")

# Maze
# env = gym.make('AntMaze_UMaze-v4', max_episode_steps=100, render_mode="human")

# Adroit Hand
# env = gym.make('AdroitHandDoor-v1', max_episode_steps=400, render_mode="human")
# env = gym.make('AdroitHandHammer-v1', max_episode_steps=400, render_mode="human")

# Franka Kitchen
env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle'], render_mode="human")





# Num of training episodes = 3
for episode in range(10):
    # Reset
    observation, info = env.reset(seed=123, options={})

    done = False
    episode_reward = 0.0
    step = 0
    while not done:
        # Agent policy that uses the observation and info
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, info = env.step(action)
        step += 1

        print("Step: {0}, Obs.: {1}, Action: {2}, Next Obs.: {3}, Reward: {4:5.2f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
            step, observation, action, next_observation, reward, terminated, truncated, info
        ))

        episode_reward += reward
        observation = next_observation
        done = terminated or truncated

    print(f"Episode: {episode + 1}, Episode Reward: {episode_reward}")

env.close()
