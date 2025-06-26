import os
import sys
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
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


class RLBenchRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        max_steps=500,
        n_obs_steps=16,
        n_action_steps=8,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
    ):
        super().__init__(output_dir)

        rotation_transformer = None
        if abs_action:
            from unified_video_action.model.common.rotation_transformer import (
                RotationTransformer,
            )

            rotation_transformer = RotationTransformer("axis_angle", "quaternion")

        # Init environment
        # TODO:
        env = 
        
        self.env = env

        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy, vis_pred_video=False, **kwargs):
        device = policy.device
        # dtype = policy.dtype
        env = self.env

        # For each eval task
        for i in range(n_tasks):

            # start rollout
            obs = env.reset()  # dict: {"agentview_image": (1, 16, 3, 128, 128)}, 
                               # here 16 is the initial frame repeated by 16 times.
                               # 16 is self.n_obs_steps
            language_goal = ""

            past_action_list = []
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False

            while not done:
                np_obs_dict = dict(obs)

                if self.past_action:
                    if len(past_action_list) > 1:  ## get 16 actions
                        np_obs_dict["past_action"] = np.concatenate(
                            past_action_list, axis=1
                        )  # (1, 16, 10)

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(
                        obs_dict,
                        language_goal=[language_goal] * obs_dict["agentview_image"].size(0),
                        **kwargs,
                    )

                # device_transfer
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

                obs, reward, done, info = env.step(env_action)
                # obs: dict: {"agentview_image": (1, 16, 3, 128, 128)},
                # each time, 8 frames in obs are updated
                # reward: [float], done: [bool], length is 1

                for i in range(len(reward)):
                    if reward[i] == 1:
                        done[i] = True

                done = np.all(done)

                # past_action = action
                past_action_list.append(action)
                if len(past_action_list) > 2:
                    past_action_list.pop(0)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            # TODO:

        # clear out video buffer
        _ = env.reset()

        # log
        # TODO:

        return log_data

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
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        return uaction
