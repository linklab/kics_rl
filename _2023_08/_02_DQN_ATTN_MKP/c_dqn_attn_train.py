import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from copy import deepcopy
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch

from _2023_08._01_DQN_MKP.a_config import env_config, dqn_config
from _2023_08._01_DQN_MKP.c_mkp_env import MkpEnv
from _2023_08._01_DQN_MKP.f_dqn_train import DQN
from _2023_08._02_DQN_ATTN_MKP.b_qnet_attn import QNetAttn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    project_home = os.path.abspath(os.path.join(current_path, os.pardir))
    if project_home not in sys.path:
        sys.path.append(project_home)

    model_dir = os.path.join(project_home, "_02_DQN_ATTN_MKP", "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    env = MkpEnv(env_config=env_config)
    test_env = deepcopy(env)

    print("*" * 100)

    use_wandb = False
    dqn = DQN(
        model=QNetAttn, model_dir=model_dir,
        env=env, test_env=test_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
