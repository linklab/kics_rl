import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import random
import torch
from matplotlib import pyplot as plt
from tensordict import TensorDict
from torchrl.envs.libs.gym import GymEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = GymEnv("CartPole-v1")
env.set_seed(0)
env.to(DEVICE)

print("Env observation_spec: \n", env.observation_spec, end="\n\n")
print("Env action_spec: \n", env.action_spec, end="\n\n")
print("Env reward_spec: \n", env.reward_spec, end="\n\n")

class Dummy_Agent:
    def get_action(self, observation):
        # observation is not used
        available_action_ids = [0, 1, 2, 3]
        action_id = random.choice(available_action_ids)
        return action_id

def main():
    print("START RUN!!!")
    agent = Dummy_Agent()
    tensordict = env.reset()
    print(tensordict)

    episode_step = 0
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_step += 1
        print("[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4:>2}, "
              "Reward: {5}, Terminated: {6}, Truncated: {7}, Info: {8}".format(
            episode_step, observation, action, ACTION_STRING_LIST[action],
            next_observation, reward, terminated, truncated, info
        ))
        observation = next_observation
        done = terminated or truncated
        time.sleep(0.5)


if __name__ == "__main__":
    main()

