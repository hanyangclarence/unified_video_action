import sys
import os
import pathlib
import subprocess
import signal
import numpy as np
import click
import torch
import dill
import random
from omegaconf import open_dict
from unified_video_action.utils.load_env import load_env_runner

def setup_env_and_xvfb():
    os.environ["COPPELIASIM_ROOT"] = "/root/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
    os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":" + os.environ["COPPELIASIM_ROOT"]
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.environ["COPPELIASIM_ROOT"]
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    display_num = 105 + gpu_id
    xvfb_cmd = ["Xvfb", f":{display_num}", "-screen", "0", "1024x768x16"]
    xvfb_proc = subprocess.Popen(xvfb_cmd)
    os.environ["DISPLAY"] = f":{display_num}"
    return xvfb_proc.pid

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("--device", default="cuda:0")
@click.option("--task", required=True, type=str)  # specify only one task
@click.option("--episode", required=True, type=int)  # specify only one episode
@click.option("--task_mode", default="policy_model")
@click.option("--cfg_pos", type=int, default=None)
@click.option("--cfg_neg", type=int, default=None)
@click.option("--server_ip", default=None, help="IP address of the server")
@click.option("--pos_neg_sample", default=False, help="Whether to use positive/negative sampling")
@click.option("--save_full_video", default=False, help="Whether to save full video of the episode")
def main(checkpoint, output_dir, device, task, episode, task_mode, cfg_pos, cfg_neg, server_ip, pos_neg_sample, save_full_video):
    xvfb_pid = setup_env_and_xvfb()

    try:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load checkpoint
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]

        # Seed setup
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        with open_dict(cfg):
            cfg.output_dir = output_dir
            cfg.task.env_runner.shape_meta = {
                "image_resolution": 256,
                "action": {"shape": [10]},
                "obs": {
                    "agentview_rgb": {
                        "shape": [3, 256, 256],
                        "type": "rgb",
                    }
                }
            }
            cfg.task.env_runner._target_ = 'unified_video_action.env_runner.rlbench_runner_client.RLBenchRunner'
            cfg.task.env_runner.start_episode = episode
            cfg.task.env_runner.eval_episodes = 1
            cfg.task.env_runner.task_start = 0
            cfg.task.env_runner.task_end = 0  # We'll manually set the task
            cfg.task.env_runner.server_ip = server_ip

        env_runner = load_env_runner(cfg, output_dir)
        env_runner.tasks = [task]  

        runner_log = env_runner.run(
            task_mode=task_mode,
            pos_neg_sample=pos_neg_sample,
            cfg_pos=cfg_pos,
            cfg_neg=cfg_neg,
            save_full_video=save_full_video
        )

        print("Runner log:", runner_log)

    finally:
        os.kill(xvfb_pid, signal.SIGTERM)


if __name__ == "__main__":
    main()