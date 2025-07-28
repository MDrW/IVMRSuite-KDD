import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


csv_files_0929 = {
    'type-1': 'test_all_metrics_ecs-v2_type_2_mlp_1_2024_09_22_01_34_07',
    'type-2': 'test_all_metrics_ecs-v2_type_3_mlp_1_2024_09_30_00_35_36',
    'type-3': 'test_all_metrics_ecs-v2_type_5_mlp_1_2024_09_28_18_45_35',
    'type-4': 'test_all_metrics_ecs-v2_type_6_mlp_1_2024_09_28_19_08_20',
    'type-5': 'test_all_metrics_ecs-v2_type_7_mlp_1_2024_09_28_19_00_56',
    'type-6': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_28_19_00_55',
    'type-7': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_09_28_19_00_42',
    'type-8': 'test_all_metrics_ecs-v2_type_10_mlp_1_2024_09_28_19_01_07',
}

csv_files_0930 = {
    'type-1': 'test_all_metrics_ecs-v2_type_2_attn_graph_1_2024_10_05_09_46_22',
    'type-2': 'test_all_metrics_ecs-v2_type_3_attn_graph_1_2024_10_06_22_28_25',
    'type-3': 'test_all_metrics_ecs-v2_type_5_attn_graph_1_2024_10_13_20_51_03',
    'type-4': 'test_all_metrics_ecs-v2_type_6_attn_graph_1_2024_10_06_22_25_58',
    'type-5': 'test_all_metrics_ecs-v2_type_7_mlp_1_2024_10_06_06_11_04',
    'type-6': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_10_06_06_11_02',
    'type-7': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_10_06_06_19_43',
    'type-8': 'test_all_metrics_ecs-v2_type_10_mlp_1_2024_10_06_06_19_52',
}

csv_files_1014 = {
    'type-1': 'test_all_metrics_ecs-v2_type_2_attn_graph_1_2024_10_16_04_42_21',
    'type-2': 'test_all_metrics_ecs-v2_type_3_attn_graph_1_2024_10_13_20_58_52',
    'type-3': 'test_all_metrics_ecs-v2_type_5_attn_graph_1_2024_10_13_20_51_03',
    'type-4': 'test_all_metrics_ecs-v2_type_6_attn_graph_1_2024_10_13_20_58_42',
    'type-5': 'test_all_metrics_ecs-v2_type_7_attn_graph_1_2024_10_16_04_32_49',
    'type-6': 'test_all_metrics_ecs-v2_type_8_attn_graph_1_2024_10_16_04_36_32',
    'type-7': 'test_all_metrics_ecs-v2_type_9_attn_graph_1_2024_10_16_04_32_51',
    'type-8': 'test_all_metrics_ecs-v2_type_10_attn_graph_1_2024_10_16_04_33_40',
}

def get_rl_mean(csv_file):
    data = pd.read_csv(f'/mnt/workspace/DRL-based-VM-Rescheduling/{csv_file}.csv')
    return np.mean(data['rl_score'])

def get_greedy_mean(csv_file):
    data = pd.read_csv(f'/mnt/workspace/DRL-based-VM-Rescheduling/{csv_file}.csv')
    return np.mean(data['greedy_score'])

def get_base_mean(csv_file):
    data = pd.read_csv(f'/mnt/workspace/DRL-based-VM-Rescheduling/{csv_file}.csv')
    return np.mean(data['base_score'])

# Prepare data for plotting
types = list(csv_files_0930.keys())
greedy_means = [get_greedy_mean(csv_files_0930[t]) for t in types]
base_means = [get_base_mean(csv_files_0930[t]) for t in types]
rl_means_0929 = [get_rl_mean(csv_files_0929[t]) for t in types]
rl_means_0930 = [get_rl_mean(csv_files_0930[t]) for t in types]
rl_means_1014 = [get_rl_mean(csv_files_1014[t]) for t in types]

# Plotting
fig, ax = plt.subplots(figsize=(20, 10))  # 进一步增加图表大小
x = np.arange(len(types))
width = 0.15  # 减小每个柱子的宽度

# 调整每个柱状图的位置
rects0 = ax.bar(x - 2*width, [m/1e8 for m in greedy_means], width, label='Greedy', alpha=0.8)
rects1 = ax.bar(x - width, [m/1e8 for m in rl_means_0929], width, label='RL-MLP(Individual)', alpha=0.8)
rects2 = ax.bar(x, [m/1e8 for m in rl_means_1014], width, label='RL-GraphAttn(Global)', alpha=0.8)
rects3 = ax.bar(x + width, [m/1e8 for m in rl_means_0930], width, label='RL-GraphAttn(Individual)', alpha=0.8)
rects4 = ax.bar(x + 2*width, [m/1e8 for m in base_means], width, label='Base', alpha=0.8)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Score (1e8)', fontsize=12)
ax.set_xlabel('Type', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(types, rotation=45, ha='right')
ax.legend(fontsize=10)

# 调整y轴的刻度
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Add value labels on top of each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=12)

autolabel(rects0)
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

fig.tight_layout()

# Save the plot
plt.savefig('/mnt/workspace/DRL-based-VM-Rescheduling/plot_res/rl_score_comparison_0930_1014.png', dpi=300, bbox_inches='tight')
print("Plot saved as /mnt/workspace/DRL-based-VM-Rescheduling/plot_res/rl_score_comparison_0930_1014.png")
