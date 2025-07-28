import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
import numpy as np

def get_tensorboard_data(path, tag):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    events = event_acc.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values

def smooth_data(values, weight=0.85):
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def plot_data(data_dict, tag, output_dir, smooth_weight=0.85):
    plt.figure(figsize=(12, 6))
    
    for label, log_dir in data_dict.items():
        steps, values = get_tensorboard_data(log_dir, tag)
        smoothed_values = smooth_data(values, smooth_weight)
        
        # Plot smoothed data
        plt.plot(steps, smoothed_values, label=f"{label} (Smoothed)")
        
        # Plot original data with lower alpha
        plt.plot(steps, values, alpha=0.3, color=plt.gca().lines[-1].get_color())

    plt.title(f"{tag} Comparison")
    plt.xlabel("Steps")
    plt.ylabel(tag)
    plt.legend()
    plt.grid(True)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"{tag.replace('/', '_')}_comparison_smoothed.png")
    plt.savefig(output_file)
    print(f"Result is saved in {output_file}")
    plt.close()

# Data dictionary
data_dict = {
    "base": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_24_00_49_56",
    # "graph": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_attn_graph_1_2024_09_25_20_14_30",
    "dense": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_25_23_31_50",
    "dense-all": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_26_19_27_25",
    "dense-2/3": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_26_19_23_57",
    "dense-0.995": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_28_20_05_35",
    # "graph_dense": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_attn_graph_1_2024_09_26_00_17_48",
    # "mean-std": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_24_23_32_33",
    "max-min-s_enhance": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_25_01_25_57",
    # "max-min": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_24_23_54_01",
    # "max-min-s_enhance_large_net": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_5_mlp_1_2024_09_25_01_30_03",
}

data_dict = {
    "base": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_2_mlp_1_2024_09_28_20_15_05",
    "graph": "/mnt/workspace/DRL-based-VM-Rescheduling/runs/new_type_ecs-v2_type_2_attn_graph_1_2024_09_30_00_55_23",
}

# Plot the data
output_dir = "/mnt/workspace/DRL-based-VM-Rescheduling/log/sand/"
plot_data(data_dict, "episode_details/episodic_score", output_dir)

# If you want to plot multiple tags, just call plot_data multiple times
# plot_data(data_dict, "Validation/Score", output_dir)
# plot_data(data_dict, "Test/Accuracy", output_dir)
