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
import click

@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to the model checkpoint")
@click.option("-o", "--output_dir", required=True, help="Output directory for results")
@click.option("--device", default="cuda:0", help="Device to use (e.g., cuda:0)")
@click.option("--task_mode", default="policy_model", help="Task mode")
@click.option("--episode_start", default=0, type=int, help="Starting episode number")
@click.option("--episodes_per_task", default=5, type=int, help="Number of episodes per task")
@click.option("--server_ip", required=True, help="IP address of the server")
@click.option("--tasks", type=str, default=None, help="Comma-separated list of tasks to run (if not specified, runs all default tasks)")
@click.option("--cfg_pos", type=int, default=None, help="Configuration for positive sampling")
@click.option("--cfg_neg", type=int, default=None, help="Configuration for negative sampling")
@click.option("--pos_neg_sample", default=False, help="Whether to use positive/negative sampling")
def main(
    checkpoint, output_dir, device, task_mode, episode_start,
    episodes_per_task, server_ip, tasks,
    cfg_pos, cfg_neg, pos_neg_sample
):
    """Run RLBench evaluation across multiple tasks and episodes."""
    
    # Use specified tasks or default task list
    if tasks:
        task_list = [task.strip() for task in tasks.split(',')]
    else:
        task_list = selected_task_list

    for task in task_list:
        for episode in range(episode_start, episode_start + episodes_per_task):
            print(f"\n>>> Running task={task}, episode={episode}...\n")

            cmd = [
                "python", "eval_rlbench/client.py",  
                "--checkpoint", checkpoint,
                "--output_dir", output_dir,
                "--task", task,
                "--episode", str(episode),
                "--device", device,
                "--task_mode", task_mode,
                "--server_ip", server_ip,
                "--pos_neg_sample", str(pos_neg_sample),
            ]
            if cfg_pos is not None:
                cmd.extend(["--cfg_pos", str(cfg_pos)])
            if cfg_neg is not None:
                cmd.extend(["--cfg_neg", str(cfg_neg)])

            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"[ERROR] Task '{task}' Episode '{episode}' failed.\n")
            else:
                print(f"[OK] Completed task '{task}' Episode '{episode}'.\n")

if __name__ == "__main__":
    main()