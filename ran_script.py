import random

import habitat_sim
import numpy as np
import torch
import torch.nn.functional as F
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ran_policy import GCNPointNavBaselinePolicy
from habitat_baselines.rl.ppo.ran_ppo_trainer import RANPPOTrainer  # RANPPOTrainer
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from PIL import Image


# A function to build configuration for PPO training
def build_PPO_config():
    # config = get_config("pointnav/ppo_pointnav.yaml")
    config = get_config(
        "/Users/mariamhegazy/CS_503_ws/habitat-lab/habitat-baselines/habitat_baselines/config/pointnav/ppo_pointnav.yaml"
    )

    # Change for REINFORCE
    OmegaConf.set_readonly(config, False)
    config.habitat_baselines.checkpoint_folder = "data/PPO_checkpoints"
    config.habitat_baselines.tensorboard_dir = "tb/PPO"
    config.habitat_baselines.num_updates = -1
    config.habitat_baselines.num_environments = 1
    config.habitat_baselines.verbose = False
    config.habitat_baselines.num_checkpoints = -1
    config.habitat_baselines.checkpoint_interval = 1000000
    config.habitat_baselines.total_num_steps = 200 * 1000
    config.habitat_baselines.force_blind_policy = True
    config.habitat.dataset.data_path = (
        "data/datasets/pointnav/simple_room/v0/{split}/empty_room.json.gz"
    )
    OmegaConf.set_readonly(config, True)
    return config


def main():
    print("A")
    config = build_PPO_config()  # Build the config for PPO
    print("B")
    # Set randomness
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)

    print("C")
    import os

    os.environ["MAGNUM_LOG"] = "quiet"
    os.environ["HABITAT_SIM_LOG"] = "quiet"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    print("D")
    # Build the trainer and start training
    trainer = RANPPOTrainer(config)
    print("E")
    trainer.train()


if __name__ == "__main__":
    main()
