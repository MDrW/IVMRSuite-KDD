import pandas as pd
import matplotlib.pyplot as plt
import glob

def process_csv(file_path):
    df = pd.read_csv(file_path)
    
    rl_mean = df['rl_score'].mean()
    greedy_mean = df['greedy_score'].mean()
    base_mean = df['base_score'].mean()
    
    return rl_mean, greedy_mean, base_mean

csv_files = {
    'type-1': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_05_00_55_22.csv',
    'type-2': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_3_mlp_1_2024_09_05_00_56_31.csv',
    'type-3': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_4_mlp_1_2024_09_05_00_56_02.csv',
    'type-4': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_5_mlp_1_2024_09_05_00_55_02.csv',
    'type-5': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_6_mlp_1_2024_09_05_05_15_03.csv',
    'type-6': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_05_00_58_04.csv',
}

all_results = []

for dataset_type, csv_file in csv_files.items():
    results = process_csv(csv_file)
    all_results.append((dataset_type, *results))

dataset_types = [result[0] for result in all_results]
rl_means = [result[1] for result in all_results]
greedy_means = [result[2] for result in all_results]
base_means = [result[3] for result in all_results]

plt.figure(figsize=(16, 10))
plt.plot(range(len(dataset_types)), rl_means, 'o-', label='RL Score', linewidth=2, markersize=10)
plt.plot(range(len(dataset_types)), greedy_means, 's-', label='Greedy Score', linewidth=2, markersize=10)
plt.plot(range(len(dataset_types)), base_means, '^-', label='Base Score', linewidth=2, markersize=10)

plt.xlabel('Dataset Types', fontsize=16)
plt.ylabel('Mean Scores', fontsize=16)
plt.title('Comparison of Mean Scores Across Dataset Types', fontsize=20)
plt.xticks(range(len(dataset_types)), dataset_types, rotation=45, ha='right', fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()
plt.grid(True)


y_min = min(min(rl_means), min(greedy_means), min(base_means))
y_max = max(max(rl_means), max(greedy_means), max(base_means))
plt.ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))


save_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/plot_csv/compare_all.png"
plt.savefig(save_path)
print(f"save plot to {save_path}")

for result in all_results:
    print(f"File: {result[0]}")
    print(f"  RL Mean: {result[1]}")
    print(f"  Greedy Mean: {result[2]}")
    print(f"  Base Mean: {result[3]}")
    print()

avg_rl_base_ratio = sum(result[1] / result[3] if result[3] != 0 else 0 for result in all_results) / len(all_results)
avg_greedy_base_ratio = sum(result[2] / result[3] if result[3] != 0 else 0 for result in all_results) / len(all_results)
avg_rl_greedy_ratio = sum(result[1] / result[2] if result[2] != 0 else 0 for result in all_results) / len(all_results)

print("Average Ratios across all files:")
print(f"  Avg RL/Base Ratio: {avg_rl_base_ratio:.4f}")
print(f"  Avg Greedy/Base Ratio: {avg_greedy_base_ratio:.4f}")
print(f"  Avg RL/Greedy Ratio: {avg_rl_greedy_ratio:.4f}")

