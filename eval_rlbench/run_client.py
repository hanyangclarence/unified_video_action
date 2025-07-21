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

import subprocess
import os

CHECKPOINT = "/root/unified_video_action/checkpoints/uva_rlbench_video_act_model/checkpoints/epoch=0065-val_action_l2_distances=0.177.ckpt"
OUTPUT_DIR = "/root/uva_rollout_DEBUG"
DEVICE = "cuda:0"
TASK_MODE = "policy_model"

episodes_per_task = 5 
server_ip = "172.31.234.23"

for task in selected_task_list:
    for episode in range(episodes_per_task):
        print(f"\n>>> Running task={task}, episode={episode}...\n")

        cmd = [
            "python", "eval_rlbench/client.py",  
            "--checkpoint", CHECKPOINT,
            "--output_dir", OUTPUT_DIR,
            "--task", task,
            "--episode", str(episode),
            "--device", DEVICE,
            "--task_mode", TASK_MODE,
            "--server_ip", server_ip,
        ]

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"[ERROR] Task '{task}' Episode '{episode}' failed.\n")
        else:
            print(f"[OK] Completed task '{task}' Episode '{episode}'.\n")