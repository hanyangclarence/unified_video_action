import os
import sys
import math
import json
import shutil
import logging
import argparse
import collections
import pathlib
import textwrap
from copy import deepcopy
import requests

import numpy as np
import torch
import h5py
import dill
import cv2
import yaml
import wandb
import quaternion
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
import json
import json_numpy

json_numpy.patch()

import wandb.sdk.data_types.video as wv
from unified_video_action.gym_util.async_vector_env import AsyncVectorEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper
from unified_video_action.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.env_runner.base_image_runner import BaseImageRunner
from unified_video_action.env.rlbench.rlbench_env import RLBenchEnv
from rlbench.backend import task as rlbench_task
from racer.rvt.utils.peract_utils import CAMERAS, IMAGE_RGB
from racer.utils.racer_utils import RLBENCH_TASKS
from racer.evaluation.policy_agent import ModelRVTAgent
from racer.evaluation.utils import START_ACTION, get_robot_delta_state, TEMPLATE_first_step, TEMPLATE_other_step

class RLBenchRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        shape_meta,
        server_ip=None,
        start_episode=0,
        eval_episodes=100,
        max_steps=500,
        n_obs_steps=16,
        n_action_steps=8,
        past_action=False,
        abs_action=False,
        fps=10,
        crf=22,
        tqdm_interval_sec=5.0,
        task_start=0,
        task_end=11,
    ):
        super().__init__(output_dir)

        rotation_transformer = None
        if abs_action:
            from unified_video_action.model.common.rotation_transformer import (
                RotationTransformer,
            )

            rotation_transformer = RotationTransformer("quaternion", "rotation_6d")

        self.tasks = None
        self.server_ip = server_ip
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        
        self.start_episode = start_episode
        self.eval_episodes = eval_episodes
        self.retry_for_InvalidActionError = 5
        self.episode_length = 50
        self.output_dir = output_dir
        self.shape_meta = shape_meta
        
        self.metric_dict = {}
        self.debug = False
    
    def env_fn(self, task_name, episode_num, episode_length):        
        return MultiStepWrapper(
                RLBenchEnv(
                    task_name=task_name,
                    dataset_root='/root/RACER/data/rlbench/test',
                    episode_length=episode_length,
                    episode_num=episode_num,
                    shape_meta=self.shape_meta,
                ),
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=self.max_steps,
        )

    def run(self, model_kwargs=None, eval_kwargs=None):
        for task_name in self.tasks:
            for episode_num in range(
                self.start_episode,
                self.start_episode + self.eval_episodes,
            ):
                log = self.eval_episode(task_name, episode_num, model_kwargs=model_kwargs, eval_kwargs=eval_kwargs)
        return log

    def eval_episode(self, task_name, episode_num, model_kwargs=None, eval_kwargs=None):
        if episode_num == 0:
            print(f"\tModel kwargs: {model_kwargs}")
            print(f"\tEval kwargs: {eval_kwargs}")

        device = 'cuda:0'
        save_dir = f"{self.output_dir}/{task_name}/{episode_num}/"
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(os.path.join(save_dir, "log.json")):
            print(f"Episode {episode_num} for task {task_name} already exists, skipping.")
            return None
        
        # start the episode
        pbar = tqdm(total=self.episode_length, desc=f"Episode {episode_num} for task {task_name} ... ", leave=False)
        env = self.env_fn(task_name, episode_num, self.episode_length)
        obs = env.reset()
        
        task_goal = env.task_goal
        print(f"task goal: {task_goal}")
        step = 0
        done = False
        success = False
        frames = []
        while not done:
            task_goal = env.task_goal
            language_goal = task_goal

            obs_dict = dict_apply(
                obs, lambda x: torch.from_numpy(x).to(device=device)
            )
            obs_dict["agentview_image"] = obs_dict.pop("agentview_rgb")
            obs_dict["agentview_image"].unsqueeze_(0)

            language_goal=[language_goal] * obs_dict["agentview_image"].size(0)

            obs_dict_np = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in obs_dict.items()
            }

            kwargs_np = {
                k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in model_kwargs.items()
            }

            if isinstance(language_goal, torch.Tensor):
                language_goal = language_goal.cpu().numpy()

            result = requests.post(
                f"http://{self.server_ip}:8000/act",
                json={"obs_dict": obs_dict_np, "language_goal": language_goal, "kwargs": kwargs_np}
            ).json()
            
            action = result["result"]["action"]  # (1, 8, 10), # 8 is self.n_action_steps
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)
            obs, reward, done, info = env.step(env_action[0])
            
            rgb_uint8 = (np.transpose(obs["agentview_rgb"], (0, 2, 3, 1)) * 255).astype(np.uint8)
            if eval_kwargs["save_full_video"]:
                frames.extend([Image.fromarray(frame) for frame in rgb_uint8[-self.n_action_steps:]])
            else:
                frames.append(Image.fromarray(rgb_uint8[-1]))

            if model_kwargs["task_mode"] == "full_dynamic_model":
                pred_video = result["result"]["video"].squeeze(0)  # (T, H, W, C)
                rollout_video = obs["agentview_rgb"][-self.n_action_steps:]  # (T, C, H, W)
                rollout_video = np.transpose((rollout_video * 255).astype(np.uint8), (0, 2, 3, 1))  # (T, H, W, C)
                
                pred_video_frames = [
                    Image.fromarray(pred_video[i]) for i in range(pred_video.shape[0])
                ]
                rollout_video_frames = [
                    Image.fromarray(rollout_video[i]) for i in range(rollout_video.shape[0])
                ]
                
                pred_video_frames[0].save(
                    os.path.join(save_dir, f"pred_video_{step}.gif"),
                    save_all=True,
                    append_images=pred_video_frames[1:],
                    duration=1000,
                    loop=0
                )
                rollout_video_frames[0].save(
                    os.path.join(save_dir, f"rollout_video_{step}.gif"),
                    save_all=True,
                    append_images=rollout_video_frames[1:],
                    duration=500,
                    loop=0
                )

            if env.is_success():
                success = True
                break
            if step >= self.episode_length:
                break
            step += 1
            pbar.update(1)
        pbar.close()
        
        log = {
            "task_name": task_name,
            "episode_num": episode_num,
            "success": success,
            "step": step,
            "language_goal": language_goal,
        }
        
        env.close()
        print(f"Episode {episode_num} for task {task_name} finished.")
        
        gif_path = os.path.join(save_dir, "agentview_rgb.gif")
        # resize the images to 128 x 128 to save space
        frames = [frame.resize((128, 128)) for frame in frames]
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,     
            loop=0           
        )
        
        dump_path = os.path.join(save_dir, "log.json")
        with open(dump_path, "w") as f:
            json.dump(log, f, indent=2)

        return log

    def undo_transform_action(self, action):
        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        # pytorch3d's convention of quaternion is wxyz, we need to convert it to xyzw for rlbench
        rot = np.concatenate(
            [rot[..., 1:], rot[..., :1]], axis=-1
        )        
        # Using 0 as default to enable collision detection for safety
        ignore_collision = np.zeros((*action.shape[:-1], 1), dtype=action.dtype)
        uaction = np.concatenate([pos, rot, gripper, ignore_collision], axis=-1)

        return uaction
      
    @staticmethod
    def _add_text_beneath_frame(frame, text):
        # append insturction below the frame
        image = frame
        if isinstance(image, Image.Image):
            image = np.array(image)
        h, w, c = image.shape
        font_size = 0.3
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        blank_image = np.zeros((90,w,c), dtype=np.uint8)

        lines = text.split('\n')  # Split the text into lines based on newline characters
        wrapped_lines = []
        char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]

        for line in lines:
            wrapped_lines.extend(textwrap.wrap(line, width=int(w / char_size[0])+8))  # Wrap each line

        y = 0
        for line in wrapped_lines:
            textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
            y += textsize[1] + 2
            x = 5
            cv2.putText(
                blank_image,
                line,
                (x, y),
                font,
                font_size,
                (255, 255, 255),
                font_thickness,
                lineType=cv2.LINE_AA,
            )
        # text_image = blank_image[0 : y + 20, 0:w]
        final = np.concatenate((image, blank_image), axis=0)
        return Image.fromarray(final)
    
    def add_text_to_frame(self, frame, text):
        return self._add_text_beneath_frame(frame, text)
