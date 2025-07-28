import pandas as pd
import matplotlib.pyplot as plt
import glob

def process_csv(file_path, metrics_df):
    df = pd.read_csv(file_path)
    
    merged_df = pd.merge(df, metrics_df, on='dataset_id', how='left')
    
    rl_mean_score = merged_df['rl_score'].mean()
    greedy_mean_score = merged_df['greedy_final_cost'].mean()
    base_mean_score = merged_df['opt_final_cost'].mean()
    
    rl_mean_time = merged_df['evaluation_time'].mean()
    greedy_mean_time = merged_df['greedy_time'].mean()
    opt_mean_time = merged_df['opt_time'].mean()
    
    rl_mean_reward = merged_df['rl_rewards'].mean()
    greedy_mean_reward = merged_df['greedy_reward'].mean()
    opt_mean_reward = merged_df['opt_reward'].mean()
    
    return (rl_mean_score, greedy_mean_score, base_mean_score,
            rl_mean_time, greedy_mean_time, opt_mean_time,
            rl_mean_reward, greedy_mean_reward, opt_mean_reward)

csv_files = {
    'type-1': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_05_00_55_22.csv',
    'type-2': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_3_mlp_1_2024_09_05_00_56_31.csv',
    'type-3': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_4_mlp_1_2024_09_05_00_56_02.csv',
    'type-4': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_5_mlp_1_2024_09_05_00_55_02.csv',
    'type-5': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_6_mlp_1_2024_09_05_05_15_03.csv',
    'type-6': '/mnt/workspace/DRL-based-VM-Rescheduling/test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_05_00_58_04.csv',
}

metrics_df = pd.read_csv('/mnt/workspace/DRL-based-VM-Rescheduling/greedy_and_opt_results.csv')

all_results = []
for dataset_type, csv_file in csv_files.items():
    results = process_csv(csv_file, metrics_df)
    all_results.append((dataset_type, *results))

dataset_types = [result[0] for result in all_results]
rl_scores = [result[1] for result in all_results]
greedy_scores = [result[2] for result in all_results]
opt_scores = [result[3] for result in all_results]
rl_times = [result[4] for result in all_results]
greedy_times = [result[5] for result in all_results]
opt_times = [result[6] for result in all_results]
rl_rewards = [result[7] for result in all_results]
greedy_rewards = [result[8] for result in all_results]
opt_rewards = [result[9] for result in all_results]

plt.figure(figsize=(16, 12))

# 1. Score (Cost) 
plt.subplot(3, 1, 1)
plt.plot(dataset_types, rl_scores, 'o-', label='RL Score', linewidth=3, markersize=10)
plt.plot(dataset_types, greedy_scores, 's--', label='Greedy Score', linewidth=3, markersize=10)
plt.plot(dataset_types, opt_scores, '^--', label='Opt Score', linewidth=3, markersize=10)
plt.title('Scores (Costs)', fontsize=24)
plt.xlabel('Dataset Types', fontsize=18)
plt.ylabel('Score (Cost)', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)

# 2. Time 
plt.subplot(3, 1, 2)
plt.plot(dataset_types, rl_times, 'o-', label='RL Time', linewidth=3, markersize=10)
plt.plot(dataset_types, greedy_times, 's--', label='Greedy Time', linewidth=3, markersize=10)
plt.plot(dataset_types, opt_times, '^--', label='Opt Time', linewidth=3, markersize=10)
plt.title('Execution Times', fontsize=24)
plt.xlabel('Dataset Types', fontsize=18)
plt.ylabel('Time (seconds)', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)

# 3. Reward 
plt.subplot(3, 1, 3)
plt.plot(dataset_types, rl_rewards, 'o-', label='RL Reward', linewidth=3, markersize=10)
plt.plot(dataset_types, greedy_rewards, 's--', label='Greedy Reward', linewidth=3, markersize=10)
plt.plot(dataset_types, opt_rewards, '^--', label='Opt Reward', linewidth=3, markersize=10)
plt.title('Rewards', fontsize=24)
plt.xlabel('Dataset Types', fontsize=18)
plt.ylabel('Rewards', fontsize=18)
plt.legend(fontsize=16)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()
save_path = "/mnt/workspace/DRL-based-VM-Rescheduling/plot_csv/compare_all_metrics.png"
plt.savefig(save_path, dpi=300)
print(f"Saved plot to {save_path}")



# 打印详细结果
for result in all_results:
    print(f"File: {result[0]}")
    print(f"  RL Score: {result[1]:.2f}")
    print(f"  Greedy Score: {result[2]:.2f}")
    print(f"  Opt Score: {result[3]:.2f}")
    print(f"  RL Time: {result[4]:.2f}")
    print(f"  Greedy Time: {result[5]:.2f}")
    print(f"  Opt Time: {result[6]:.2f}")
    print(f"  RL Reward: {result[7]:.2f}")
    print(f"  Greedy Reward: {result[8]:.2f}")
    print(f"  Opt Reward: {result[9]:.2f}")
    print()

# 计算平均比率
def safe_divide(a, b):
    return a / b if b != 0 else 0

# Score ratios
avg_rl_opt_score_ratio = sum(safe_divide(result[1], result[3]) for result in all_results) / len(all_results)
avg_greedy_opt_score_ratio = sum(safe_divide(result[2], result[3]) for result in all_results) / len(all_results)
avg_rl_greedy_score_ratio = sum(safe_divide(result[1], result[2]) for result in all_results) / len(all_results)

# Time ratios
avg_rl_opt_time_ratio = sum(safe_divide(result[4], result[6]) for result in all_results) / len(all_results)
avg_greedy_opt_time_ratio = sum(safe_divide(result[5], result[6]) for result in all_results) / len(all_results)
avg_rl_greedy_time_ratio = sum(safe_divide(result[4], result[5]) for result in all_results) / len(all_results)

# Reward ratios
avg_rl_opt_reward_ratio = sum(safe_divide(result[7], result[9]) for result in all_results) / len(all_results)
avg_greedy_opt_reward_ratio = sum(safe_divide(result[8], result[9]) for result in all_results) / len(all_results)
avg_rl_greedy_reward_ratio = sum(safe_divide(result[7], result[8]) for result in all_results) / len(all_results)

print("Average Ratios across all files:")
print("Scores (lower is better):")
print(f"  Avg RL/Opt Score Ratio: {avg_rl_opt_score_ratio:.4f}")
print(f"  Avg Greedy/Opt Score Ratio: {avg_greedy_opt_score_ratio:.4f}")
print(f"  Avg RL/Greedy Score Ratio: {avg_rl_greedy_score_ratio:.4f}")
print()
print("Execution Times:")
print(f"  Avg RL/Opt Time Ratio: {avg_rl_opt_time_ratio:.4f}")
print(f"  Avg Greedy/Opt Time Ratio: {avg_greedy_opt_time_ratio:.4f}")
print(f"  Avg RL/Greedy Time Ratio: {avg_rl_greedy_time_ratio:.4f}")
print()
print("Rewards (higher is better):")
print(f"  Avg RL/Opt Reward Ratio: {avg_rl_opt_reward_ratio:.4f}")
print(f"  Avg Greedy/Opt Reward Ratio: {avg_greedy_opt_reward_ratio:.4f}")
print(f"  Avg RL/Greedy Reward Ratio: {avg_rl_greedy_reward_ratio:.4f}")
