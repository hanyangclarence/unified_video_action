import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
from omegaconf import open_dict
from omegaconf import OmegaConf
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner

if "DEBUG" in os.environ and os.environ["DEBUG"] == "1":
    import pdb
    pdb.set_trace()

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-s", "--start_episode", default=0)
@click.option("-e", "--eval_episodes", default=100)
def main(checkpoint, output_dir, device, start_episode, eval_episodes):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    
    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        # Missing shape meta in ckpt, manually set it
        cfg.task.env_runner.shape_meta = {
            "image_resolution": 256,
            "action": {
                "shape": [10],
            },
            "obs": {
                "agentview_rgb": {
                    "shape": [3, 256, 256],
                    "type": "rgb",
                }
            }
        }
        cfg.task.env_runner.start_episode = start_episode
        cfg.task.env_runner.eval_episodes = eval_episodes
        
    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    
    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    env_runners = load_env_runner(cfg, output_dir)

    if "libero" in cfg.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)
    print("Runner log:", runner_log)

if __name__ == "__main__":
    main()
