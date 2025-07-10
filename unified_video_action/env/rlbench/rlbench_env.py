from numpy import ndarray
from PIL import Image
import numpy as np
import cv2
from gym import spaces

from rlbench.backend.const import *
from rlbench.backend.utils import task_file_to_task_class

from racer.evaluation.utils import ROLLOUT_IMAGE_SIZE, CustomRLRenchEnv2
from racer.peract.helpers import utils
from racer.utils.racer_utils import CAMERAS
from racer.evaluation.utils_cogkit import MoveArmThenGripper2, Discrete2, EndEffectorPoseViaPlanning2
from yarr.agents.agent import ActResult
from yarr.utils.transition import Transition
from racer.evaluation.utils import STAND_POSE_ACTION



class RLBenchEnv:
    def __init__(
        self,  
        task_name: str,
        episode_num: int,
        dataset_root: str,
        episode_length: int=30,
        record_every_n: int = 5, # -1 means no recording
        resolution: int=ROLLOUT_IMAGE_SIZE,
        record_queue=None,
        shape_meta: dict = None,
        never_terminal=False,
        unseen_task=False,
    ):
        self.task_name = task_name
        self.dataset_root = dataset_root
        self.episode_length = episode_length
        self.record_every_n = record_every_n
        self.record_queue = record_queue
        self.never_terminal = never_terminal
        self.unseen_task = unseen_task
        self.shape_meta = shape_meta
        self.episode_num = episode_num
        
        action_shape = shape_meta["action"]["shape"]
        action_space = spaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32)
        self.action_space = action_space
        
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = value["shape"]
            min_value, max_value = -1, 1
            if key.endswith("rgb"):
                min_value, max_value = 0, 1

        this_space = spaces.Box(
            low=min_value, high=max_value, shape=shape, dtype=np.float32
        )
        observation_space[key] = this_space
        self.observation_space = observation_space

        self.setup_env(resolution)

        self.last_action = None
        
    def reset(self) -> dict:
        obs = {}
        obs_dict, _ = self.env.reset_to_demo(self.episode_num, not_load_image=True)
        obs = self.get_observation(obs_dict)
        return obs
    
    def setup_env(self, resolution):
        camera_resolution = [resolution, resolution]
        obs_config = utils.create_obs_config(CAMERAS, camera_resolution, method_name="")

        gripper_mode = Discrete2()
        arm_action_mode = EndEffectorPoseViaPlanning2()
        action_mode = MoveArmThenGripper2(arm_action_mode, gripper_mode)
        self.env = CustomRLRenchEnv2(
            record_queue=self.record_queue,
            task_class=task_file_to_task_class(self.task_name),
            observation_config=obs_config,
            action_mode=action_mode,
            dataset_root=self.dataset_root,
            episode_length=self.episode_length,
            headless=True,
            time_in_state=True,
            include_lang_goal_in_obs=True,
            record_every_n=self.record_every_n,
            never_terminal=self.never_terminal,
            unseen_task=self.unseen_task,
        )
        self.env.eval = True
        self.env.launch()
    
    def set_new_task(self, task_name: str):
        self.env.set_new_task(task_name)
        self.task_name = task_name
    
    def set_new_dataset(self, dataset_root: str):
        self.env._rlbench_env._dataset_root = dataset_root
    
    @property
    def task_goal(self):
        return self.env._lang_goal

    def get_observation(self, obs_dict) -> dict:
        obs = {}
        resized = np.stack([
            cv2.resize(obs_dict['front_rgb'][c], (256, 256), interpolation=cv2.INTER_LINEAR)
            for c in range(3)
        ], axis=0)
        # Normalize to [0.0, 1.0] and convert to float32
        obs['agentview_rgb'] = resized.astype(np.float32) / 255.0
        return obs
    
    def step(self, action: ndarray) -> Transition:
        # action is (9, ) array, 3 for pose, 4 for quaternion, 1 for gripper, 1 for ignore_collision
        wrap_action = ActResult(action=action)            
        transition = self.env.step(wrap_action) # get Transition(obs, reward, terminal, info, summaries)    
        if transition.info['error_status'] == "error": # sometimes RLbench throws strange error
            print(f"Error: action was {action}")
            if self.task_name in ["put_item_in_drawer"]: 
                transition = self.env.step(ActResult(action=STAND_POSE_ACTION))
            if self.task_name in ["open_drawer"] and self.last_action is not None: 
                action[0] = (self.last_action[0] + action[0])/2
                action[2] = (self.last_action[2] + action[2])/2
                transition = self.env.step(ActResult(action=action))
        if isinstance(transition, tuple):
            transition = transition[0]
        self.transition = transition
        self.last_action = action
        done = transition.terminal
        obs_dict = transition.observation
        obs = self.get_observation(obs_dict)
        reward = transition.reward
        info = transition.info
        
        return obs, reward, done, info
    
    def is_success(self) -> bool:
        # always called when simulation ends
        score = self.transition.reward
        return True if score == 100.0 else False
        
    def close(self):
        self.env.shutdown()
    
    def get_video_frames(self, res=128, return_pil=True):
        ret = []
        for fra in self.env._recorded_images:
            if fra.shape[0] == 3:
                fra = fra.transpose(1, 2, 0)
            fra = Image.fromarray(fra).resize((res, res))
            if return_pil:
                ret.append(fra)
            else:
                ret.append(np.array(fra))
        self.env._recorded_images.clear()
        return ret
    