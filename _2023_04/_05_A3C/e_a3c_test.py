# https://gymnasium.farama.org/environments/classic_control/pendulum/
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import torch
import torch.multiprocessing as mp

from c_actor_and_critic import MODEL_DIR, Actor


def test(env, actor, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            # action = actor.get_action(observation)
            action = actor.get_action(observation, exploration=False)

            next_observation, reward, terminated, truncated, _ = env.step(action * 2)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))


# def worker_process(global_actor, env_name, num_episodes):
#     env = gym.make(env_name, render_mode="human")
#     local_actor = Actor(n_features=3, n_actions=1)
#     local_actor.load_state_dict(global_actor.state_dict())
#
#     test(env, local_actor, num_episodes=num_episodes)
#
#     env.close()


def main_play(num_episodes, env_name):
    env = gym.make(env_name, render_mode="human")

    global_actor = Actor(n_features=3, n_actions=1)
    model_params = torch.load(os.path.join(MODEL_DIR, "a3c_{0}_latest.pth".format(env_name)))
    global_actor.load_state_dict(model_params)

    test(env, global_actor, num_episodes=num_episodes)

    env.close()

    # global_actor = Actor(n_features=3, n_actions=1)
    # model_params = torch.load(os.path.join(MODEL_DIR, "a3c_{0}_latest.pth".format(env_name)))
    # global_actor.load_state_dict(model_params)
    #
    # processes = []
    #
    # for _ in range(num_workers):
    #     p = mp.Process(target=worker_process, args=(global_actor, env_name, num_episodes))
    #     p.start()
    #     processes.append(p)
    #
    # for p in processes:
    #     p.join()


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "Pendulum-v1"
    # NUM_WORKERS = 3  # Number of worker processes for asynchronous execution
    # main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME, num_workers=NUM_WORKERS)
    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)