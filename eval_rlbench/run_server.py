import sys
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
import logging
import traceback
from omegaconf import open_dict
from omegaconf import OmegaConf
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from unified_video_action.common.pytorch_util import dict_apply

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union


if "DEBUG" in os.environ and os.environ["DEBUG"] == "1":
    import pdb
    pdb.set_trace()

@dataclass
class ServerConfig:
    checkpoint: str
    output_dir: str
    device: str = "cuda:0"
    host: str = "0.0.0.0"
    port: int = 8000

class PolicyServer:
    def __init__(self, config: ServerConfig):
        """
        A server for policy models; exposes `/act` to predict an action for a given observation.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize the policy
        self._load_policy()
        print("PolicyServer initialized.")

    def _load_policy(self):
        """Load the policy from checkpoint"""
        # load checkpoint
        payload = torch.load(open(self.config.checkpoint, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        with open_dict(cfg):
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

        # configure workspace
        cls = hydra.utils.get_class(cfg.model._target_)
        workspace = cls(cfg, output_dir=self.config.output_dir)
        workspace: BaseWorkspace

        print("Loaded checkpoint from %s" % self.config.checkpoint)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        self.policy = workspace.ema_model
        self.policy.to(self.device)
        self.policy.eval()

        import json_numpy
        json_numpy.patch()
        
        self.cfg = cfg

    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        try:
            obs_dict, language_goal, kwargs = payload["obs_dict"], payload["language_goal"], payload["kwargs"]
            
            obs_dict = {
                k: torch.from_numpy(v).float().cuda() if isinstance(v, np.ndarray) else v
                for k, v in obs_dict.items()
            }

            with torch.no_grad():
                action_dict = self.policy.predict_action(
                    obs_dict,
                    language_goal=language_goal,
                    **kwargs,
                )
                result = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

            result = {
                "result": result,
            }

            return JSONResponse(result)
            
        except Exception as e:
            logging.error(traceback.format_exc())
            logging.warning(
                f"Your request threw an error: {str(e)}; make sure your request complies with the expected format:\n"
                "{'observation': {...}}\n"
            )
            return JSONResponse({"error": str(e), "success": False})

    def run(self) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
def main(checkpoint, output_dir, device, host, port):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = ServerConfig(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        host=host,
        port=port
    )
    server_instance = PolicyServer(config)
    server_instance.run()

if __name__ == "__main__":
    main()