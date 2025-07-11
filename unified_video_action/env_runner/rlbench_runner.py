selected_task_list = [
    "reach_and_drag",      
    "put_item_in_drawer",          
    "turn_tap",                       
    "slide_block_to_color_target",    
    "open_drawer",                    
    "place_shape_in_shape_sorter",    
    "push_buttons",                   
    "close_jar",                      
    "place_wine_at_rack_location",    
    "insert_onto_square_peg",         
    "meat_off_grill",                 
]

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

import wandb.sdk.data_types.video as wv
from unified_video_action.gym_util.async_vector_env import AsyncVectorEnv
from unified_video_action.gym_util.multistep_wrapper import MultiStepWrapper
from unified_video_action.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)

from unified_video_action.policy.base_image_policy import BaseImagePolicy
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
    ):
        super().__init__(output_dir)

        rotation_transformer = None
        if abs_action:
            from unified_video_action.model.common.rotation_transformer import (
                RotationTransformer,
            )

            rotation_transformer = RotationTransformer("quaternion", "rotation_6d")

        self.tasks = selected_task_list
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
        self.episode_length = 25
        self.output_dir = output_dir
        
        self.metric_dict = {}
        self.debug = False
        
        def env_fn(task_name, episode_num, episode_length):
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RLBenchEnv(
                        task_name=task_name,
                        dataset_root='/root/RACER/data/rlbench/test',
                        episode_length=episode_length,
                        episode_num=episode_num,
                        shape_meta=shape_meta,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=2
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        self.env_fn = env_fn

    def run(self, policy: BaseImagePolicy, vis_pred_video=False, **kwargs):
        device = policy.device
        for task_name in self.tasks:
            for episode_num in range(
                self.start_episode,
                self.start_episode + self.eval_episodes,
            ):
                log = self.eval_episode(task_name, episode_num, policy, **kwargs)
        return log
    
    def eval_episode(self, task_name, episode_num, policy, **kwargs):
        device = policy.device
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
        policy.reset()
        while not done:
            task_goal = env.task_goal
            language_goal = task_goal

            obs_dict = dict_apply(
                obs, lambda x: torch.from_numpy(x).to(device=device)
            )
            obs_dict["agentview_image"] = obs_dict.pop("agentview_rgb")
            obs_dict["agentview_image"].unsqueeze_(0) 
            
            # run policy
            with torch.no_grad():
                action_dict = policy.predict_action(
                    obs_dict,
                    language_goal=[language_goal]*obs_dict["agentview_image"].size(0),
                    **kwargs,
                )
            
            np_action_dict = dict_apply(
                action_dict, lambda x: x.detach().to("cpu").numpy()
            )

            action = np_action_dict["action"]  # (1, 8, 10), # 8 is self.n_action_steps
            if not np.all(np.isfinite(action)):
                print(action)
                raise RuntimeError("Nan or Inf action")

            # step env
            env_action = action
            if self.abs_action:
                env_action = self.undo_transform_action(action)
            obs, reward, done, info = env.step(env_action[0])
            
            rgb_uint8 = (np.transpose(obs['agentview_rgb'][0], (1, 2, 0)) * 255).astype(np.uint8)
            frames.append(Image.fromarray(rgb_uint8))
            
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
