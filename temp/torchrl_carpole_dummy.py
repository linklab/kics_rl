import torch
import tqdm
from functorch import vmap
from matplotlib import pyplot as plt
from tensordict import TensorDict
from tensordict.nn import get_functional
from torch import nn
from torchrl.collectors import MultiaSyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    Compose,
    GrayScale,
    ObservationNorm,
    Resize,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.utils import set_exploration_mode, step_mdp
from torchrl.modules import DuelingCnnDQNet, EGreedyWrapper, QValueActor

config = {
    "lr": 0.002,            # the learning rate of the optimizer
    "betas": (0.9, 0.999),  # the beta parameters of Adam
    "gamma": 0.99,          # gamma decay factor
    "lmbda": 0.95,          # lambda decay factor (see second the part with TD(lambda)

    # total frames collected in the environment.
    # In other implementations, the user defines a maximum number of episodes.
    # This is harder to do with our data collectors since they return batches of N collected frames, where N is a constant.
    # However, one can easily get the same restriction on number of episodes by breaking the training loop when a certain number
    # episodes has been collected.
    "total_frames": 500,

    "init_random_frames": 100,          # Random frames used to initialize the replay buffer.
    "frames_per_batch": 32,             # Frames in each batch collected.
    "n_optim": 4,                       # Optimization steps per batch collected
    "batch_size": 32,                   # Frames sampled from the replay buffer at each optimization step
    "buffer_size": min(500, 100000),    # Size of the replay buffer in terms of frames
    "n_workers": 1,                     # Number of environments run in parallel in each data collector
    
    "device": "cuda:0" if torch.cuda.device_count() > 0 else "cpu",
    
    # Smooth target network update decay parameter.
    # This loosely corresponds to a 1/(1-tau) interval with hard target network update
    "tau": 0.005,
    
    # Initial and final value of the epsilon factor in Epsilon-greedy exploration
    # notice that since our policy is deterministic exploration is crucial
    "eps_greedy_val": 0.1,
    "eps_greedy_val_env": 0.05,
    
    # To speed up learning, we set the bias of the last layer of our value network to a predefined value
    "init_bias": 20.0,
}


def make_env(parallel=False, pixel=False):
    if parallel:
        if pixel:
            base_env = ParallelEnv(
                config["n_workers"],
                EnvCreator(lambda: GymEnv("CartPole-v1", from_pixels=True, pixels_only=True, device=config["device"])),
            )
        else:
            base_env = ParallelEnv(
                config["n_workers"],
                EnvCreator(lambda: GymEnv("CartPole-v1", device=config["device"])),
            )
    else:
        if pixel:
            base_env = GymEnv("CartPole-v1", from_pixels=True, pixels_only=True, device=config["device"])
        else:
            base_env = GymEnv("CartPole-v1", device=config["device"])

    if pixel:
        env = TransformedEnv(
            base_env,
            Compose(
                ToTensorImage(),
                GrayScale(),
                Resize(64, 64),
                ObservationNorm(in_keys=["pixels"], loc=0, scale=1, standard_normal=True),
                CatFrames(4, in_keys=["pixels"], dim=-3),
            ),
        )
    else:
        env = base_env

    return env


dummy_env = make_env()
