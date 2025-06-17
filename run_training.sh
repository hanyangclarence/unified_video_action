export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# accelerate launch --num_processes=1 train.py \
#     --config-dir=. \
#     --config-name=uva_libero10.yaml \
#     model.policy.action_model_params.predict_action=False \
#     model.policy.selected_training_mode=video_model \
#     model.policy.optimizer.learning_rate=1e-4 \
#     logging.project=uva \
#     hydra.run.dir="checkpoints/uva_libero10_video_model"

torchrun --nnode=1 --nproc_per_node=1 train.py --config-dir=. --config-name=uva_libero10.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    model.policy.optimizer.learning_rate=1e-4 \
    logging.project=uva \
    hydra.run.dir="checkpoints/uva_libero10_video_model"