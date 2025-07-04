export WANDB_API_KEY=28b3c634497c0dc6c16767729d4719b1012a94f2
export DEBUG=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# accelerate launch --num_processes=8 train.py \
#         --config-dir=. \
#         --config-name=uva_rlbench_pn.yaml \
#         model.policy.action_model_params.predict_action=False \
#         model.policy.selected_training_mode=video_model \
#         model.policy.optimizer.learning_rate=1e-4 \
#         logging.project=uva_pn \
#         hydra.run.dir="checkpoints/uva_rlbench_pn_video_model"


accelerate launch --num_processes=8 train.py \
        --config-dir=. \
        --config-name=uva_rlbench_pn.yaml \
        model.policy.action_model_params.predict_action=True \
        model.policy.autoregressive_model_params.pretrained_model_path="checkpoints/uva_rlbench_pn_video_model/checkpoints/best.ckpt" \
        model.policy.optimizer.learning_rate=1e-4 \
        logging.project=uva_pn \
        hydra.run.dir="checkpoints/uva_rlbench_pn_video_act_model" \
        checkpoint.topk.monitor_key="val_action_l2_distances" \
        checkpoint.topk.format_str='epoch={epoch:04d}-val_action_l2_distances={val_action_l2_distances:.3f}.ckpt'