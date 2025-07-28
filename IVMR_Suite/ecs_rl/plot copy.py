import pandas as pd
import matplotlib.pyplot as plt


result_csv = 'test_ecs-v2_type_7_mlp_1_2024_09_04_17_58_00'
data = pd.read_csv(f'/mnt/workspace/DRL-based-VM-Rescheduling/{result_csv}.csv')

row_indices = data.index 
env_ids = data['dataset_id']
greedy_rewards = data['greedy_rewards']
rl_rewards = data['rl_rewards']
base_rewards = data['base_rewards']
greedy_scores = data['greedy_score']
rl_scores = data['rl_score']
base_scores = data['base_score']

fig, axs = plt.subplots(2, 1, figsize=(10, 12))

axs[0].bar(row_indices, greedy_rewards, label='Greedy Rewards', alpha=0.6, width=0.2, align='center')
axs[0].bar(row_indices + 0.2, rl_rewards, label='RL Rewards', alpha=0.6, width=0.2, align='center')
axs[0].bar(row_indices + 0.4, base_rewards, label='Base Rewards', alpha=0.6, width=0.2, align='center')
axs[0].set_title('Rewards Metric')
axs[0].set_xlabel('Dataset Index')
axs[0].set_ylabel('Rewards')
axs[0].legend()
axs[0].grid()

axs[1].bar(row_indices, greedy_scores, label='Greedy Score', alpha=0.6, width=0.2, align='center')
axs[1].bar(row_indices + 0.2, rl_scores, label='RL Score', alpha=0.6, width=0.2, align='center')
axs[1].bar(row_indices + 0.4, base_scores, label='Base Score', alpha=0.6, width=0.2, align='center')
axs[1].set_title('Scores Metric')
axs[1].set_xlabel('Dataset Index')
axs[1].set_ylabel('Scores')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
save_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/plot_res/{result_csv}.png"
plt.savefig(save_path)
print(f"save plot to {save_path}")
