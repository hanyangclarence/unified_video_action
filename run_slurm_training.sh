#!/bin/bash
#SBATCH --job-name=yh
#SBATCH -o slurm_output/tr_%j.out
#SBATCH -e slurm_output/tr_%j.err
#SBATCH --mem=500G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:8

cd /gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/embodied_o1/unified_video_action
conda init
conda activate uva

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2

export MUJOCO_PY_MUJOCO_PATH=/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/embodied_o1/mujoco/mujoco210/bin

LD_LIBRARY_PATH="/usr/lib/nvidia:$MUJOCO_PY_MUJOCO_PATH:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" torchrun --nnode=1 --nproc_per_node=8 train.py --config-dir=. --config-name=uva_rlbench.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_rlbench_video_model"