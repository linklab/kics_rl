import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%7.3f" % x))

# https://gymnasium.farama.org/environments/atari/breakout/
env = gym.make("ALE/Breakout-v5", render_mode="human")

def observation_and_action_space_info():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print("[observation_space]")
    print("env.observation_space: ", env.observation_space)
    print("env.observation_space.high:", env.observation_space.high)
    print("env.observation_space.low:", env.observation_space.low)
    print("env.observation_space.shape:", env.observation_space.shape)
    for i in range(1):
        print(env.observation_space.sample())
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print("[action_space]")
    print(env.action_space)
    print("env.action_space.n:", env.action_space.n)
    print("env.action_space.shape:", env.action_space.shape)
    for i in range(10):
        print(env.action_space.sample(), end=" ")


if __name__ == "__main__":
    observation_and_action_space_info()
