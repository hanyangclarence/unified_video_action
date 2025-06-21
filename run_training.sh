export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export WANDB_MODE=disabled
export DEBUG=1

LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" torchrun --nnode=1 --nproc_per_node=1 train.py --config-dir=. --config-name=uva_rlbench.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_rlbench_video_model"