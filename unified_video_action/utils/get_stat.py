import os
import json
from collections import defaultdict

root = "/root/uva_rollout"
eval_log_path = os.path.join(root, "evaluation_log.json")

# Initialize counters
task_success = defaultdict(list)

for task_name in os.listdir(root):
    task_path = os.path.join(root, task_name)
    if not os.path.isdir(task_path):
        continue
    for episode in os.listdir(task_path):
        episode_path = os.path.join(task_path, episode)
        if not os.path.isdir(episode_path):
            continue
        log_path = os.path.join(episode_path, "log.json")
        if not os.path.exists(log_path):
            continue
        with open(log_path, "r") as f:
            log = json.load(f)
            success = log.get("success", False)
            task_success[task_name].append(success)

task_stats = {}
total_successes = 0
total_failures = 0

print("##### Average success per task: \n")
for task, results in task_success.items():
    successes = sum(results)
    failures = len(results) - successes
    total_successes += successes
    total_failures += failures
    success_rate = successes / len(results)
    print(f"{task}: {success_rate:.2f}")
    task_stats[task] = {
        "success_rate": success_rate,
        "success_count": successes,
        "failure_count": failures,
        "total": len(results)
    }

overall_total = total_successes + total_failures
overall_success_rate = total_successes / overall_total if overall_total > 0 else 0.0
print(f"\n##### Overall success: {overall_success_rate:.2f} \n")

eval_log = {
    "task_metrics": task_stats,
    "overall_metrics": {
        "overall_success_rate": overall_success_rate,
        "total_successes": total_successes,
        "total_failures": total_failures,
        "total_episodes": overall_total
    }
}

with open(eval_log_path, "w") as f:
    json.dump(eval_log, f, indent=2)

print(f"Evaluation log saved to: {eval_log_path}")