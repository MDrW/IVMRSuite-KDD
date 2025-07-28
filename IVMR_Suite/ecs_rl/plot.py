import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # 0923
# csv_files = {
#     'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
#     'type-2': 'test_all_metrics_ecs-v2_type_2_mlp_1_2024_09_22_01_34_07',
#     'type-3': 'test_all_metrics_ecs-v2_type_3_mlp_1_2024_09_22_01_32_06',
#     'type-4': 'test_all_metrics_ecs-v2_type_4_mlp_1_2024_09_22_01_37_01',
#     'type-5': 'test_all_metrics_ecs-v2_type_5_mlp_1_2024_09_22_18_20_17',
#     'type-6': 'test_all_metrics_ecs-v2_type_6_mlp_1_2024_09_22_18_23_22',
#     'type-7': 'test_all_metrics_ecs-v2_type_7_mlp_1_2024_09_22_20_29_53',
#     'type-8': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_22_20_29_15',
#     'type-9': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_09_22_19_54_30',
#     'type-10': 'test_all_metrics_ecs-v2_type_10_mlp_1_2024_09_22_20_02_01',
# }

# #0925
# csv_files = {
#     'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
#     'type-2': 'test_all_metrics_ecs-v2_type_2_mlp_1_2024_09_22_01_34_07',
#     'type-3': 'test_all_metrics_ecs-v2_type_3_mlp_0_2024_09_25_00_52_48',
#     'type-4': 'test_all_metrics_ecs-v2_type_4_mlp_0_2024_09_25_01_16_01',
#     'type-5': 'test_all_metrics_ecs-v2_type_5_mlp_0_2024_09_25_01_12_25',
#     'type-6': 'test_all_metrics_ecs-v2_type_6_mlp_0_2024_09_25_01_13_06',
#     'type-7': 'test_all_metrics_ecs-v2_type_7_mlp_0_2024_09_25_01_13_32',
#     'type-8': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_22_20_29_15',
#     'type-9': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_09_22_19_54_30',
#     'type-10': 'test_all_metrics_ecs-v2_type_10_mlp_0_2024_09_25_00_56_02',
# }

#0929
# csv_files = {
    # 'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
    # 'type-2': 'test_all_metrics_ecs-v2_type_2_mlp_1_2024_09_22_01_34_07',
    # 'type-3': 'test_all_metrics_ecs-v2_type_3_mlp_1_2024_09_30_00_35_36',
    # 'type-4': 'test_all_metrics_ecs-v2_type_4_mlp_0_2024_09_25_01_16_01',
    # 'type-5': 'test_all_metrics_ecs-v2_type_5_mlp_1_2024_09_28_18_45_35',
    # 'type-6': 'test_all_metrics_ecs-v2_type_6_mlp_1_2024_09_28_19_08_20',
    # 'type-7': 'test_all_metrics_ecs-v2_type_7_mlp_1_2024_09_28_19_00_56',
    # 'type-8': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_28_19_00_55',
    # 'type-9': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_09_28_19_00_42',
    # 'type-10': 'test_all_metrics_ecs-v2_type_10_mlp_1_2024_09_28_19_01_07',
# }

# 0930
csv_files = {
    # 'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
    'type-2': 'test_all_metrics_ecs-v2_type_2_attn_graph_1_2024_10_05_09_46_22',
    'type-3': 'test_all_metrics_ecs-v2_type_3_attn_graph_1_2024_10_06_22_28_25',
    # 'type-4': 'test_all_metrics_ecs-v2_type_4_mlp_0_2024_09_25_01_16_01',
    # 'type-5': 'test_all_metrics_ecs-v2_type_5_attn_graph_1_2024_10_06_22_36_22',
    'type-5': 'test_all_metrics_ecs-v2_type_5_attn_graph_1_2024_10_13_20_51_03',
    'type-6': 'test_all_metrics_ecs-v2_type_6_attn_graph_1_2024_10_06_22_25_58',
    'type-7': 'test_all_metrics_ecs-v2_type_7_mlp_1_2024_10_06_06_11_04',
    'type-8': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_10_06_06_11_02',
    'type-9': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_10_06_06_19_43',
    'type-10': 'test_all_metrics_ecs-v2_type_10_mlp_1_2024_10_06_06_19_52',
}

# 1014
csv_files = {
    # 'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
    'type-2': 'test_all_metrics_ecs-v2_type_2_attn_graph_1_2024_10_16_04_42_21',
    'type-3': 'test_all_metrics_ecs-v2_type_3_attn_graph_1_2024_10_13_20_58_52',
    # 'type-4': 'test_all_metrics_ecs-v2_type_4_attn_graph_1_2024_10_16_04_49_44',
    'type-5': 'test_all_metrics_ecs-v2_type_5_attn_graph_1_2024_10_13_20_51_03',
    'type-6': 'test_all_metrics_ecs-v2_type_6_attn_graph_1_2024_10_13_20_58_42',
    'type-7': 'test_all_metrics_ecs-v2_type_7_attn_graph_1_2024_10_16_04_32_49',
    'type-8': 'test_all_metrics_ecs-v2_type_8_attn_graph_1_2024_10_16_04_36_32',
    'type-9': 'test_all_metrics_ecs-v2_type_9_attn_graph_1_2024_10_16_04_32_51',
    'type-10': 'test_all_metrics_ecs-v2_type_10_attn_graph_1_2024_10_16_04_33_40',
}

# csv_files = {
#     'type-1': 'test_all_metrics_ecs-v2_type_1_mlp_1_2024_09_22_01_51_51',
#     'type-2': 'test_all_metrics_ecs-v2_type_2_mlp_1_2024_09_22_01_34_07',
#     'type-3': 'test_all_metrics_ecs-v2_type_3_mlp_0_2024_09_24_04_24_10',
#     'type-4': 'test_all_metrics_ecs-v2_type_4_mlp_0_2024_09_24_18_47_44',
#     'type-5': 'test_all_metrics_ecs-v2_type_5_mlp_0_2024_09_24_04_41_42',
#     'type-6': 'test_all_metrics_ecs-v2_type_6_mlp_0_2024_09_24_00_52_47',
#     'type-7': 'test_all_metrics_ecs-v2_type_7_mlp_0_2024_09_24_04_12_05',
#     'type-8': 'test_all_metrics_ecs-v2_type_8_mlp_1_2024_09_22_20_29_15',
#     'type-9': 'test_all_metrics_ecs-v2_type_9_mlp_1_2024_09_22_19_54_30',
#     'type-10': 'test_all_metrics_ecs-v2_type_10_mlp_0_2024_09_24_18_46_05',
# }

if False:
    all_dataframes = []

    for test_type, result_csv in csv_files.items():
        file_path = f'/mnt/workspace/DRL-based-VM-Rescheduling/{result_csv}.csv'
        df = pd.read_csv(file_path)
        df['test_type'] = test_type  
        all_dataframes.append(df)


    combined_df = pd.concat(all_dataframes, ignore_index=True)


    combined_df.to_csv('/mnt/workspace/DRL-based-VM-Rescheduling/1007_rl_results.csv', index=False)

    print(combined_df.head())

    print(f"Total number of rows: {len(combined_df)}")
    print(f"Columns: {combined_df.columns.tolist()}")
else:
    summary_data = []

    for test_type, result_csv in csv_files.items():
        data = pd.read_csv(f'/mnt/workspace/DRL-based-VM-Rescheduling/{result_csv}.csv')

        row_indices = data.index 
        env_ids = data['dataset_id']
        greedy_scores = data['greedy_score']
        rl_scores = data['rl_score']
        base_scores = data['base_score']



        greedy_mean = np.mean(greedy_scores)
        rl_mean = np.mean(rl_scores)
        base_mean = np.mean(base_scores)

        rl_better_count = np.sum(rl_scores < greedy_scores) + np.sum(rl_scores == greedy_scores)
        rl_better_ratio = rl_better_count / len(rl_scores)

        summary_data.append({
            'Type': test_type,
            'Greedy Mean': greedy_mean,
            'RL Mean': rl_mean,
            'Base Mean': base_mean,
            'RL Better Ratio': rl_better_ratio
        })

        plt.figure(figsize=(12, 6))

        plt.bar(row_indices, greedy_scores, label='Greedy Score', alpha=0.6, width=0.2, align='center')
        plt.bar(row_indices + 0.2, rl_scores, label='RL Score', alpha=0.6, width=0.2, align='center')
        plt.bar(row_indices + 0.4, base_scores, label='Base Score', alpha=0.6, width=0.2, align='center')

        plt.title(test_type, fontsize=16)
        plt.xlabel('Dataset Index', fontsize=12)
        plt.ylabel('Scores', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        save_path = f"/mnt/workspace/DRL-based-VM-Rescheduling/plot_res/1007_{result_csv}_scores.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")



    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.set_index('Type')
    summary_df = summary_df.round(4)  


    summary_save_path = os.path.join('/mnt/workspace/DRL-based-VM-Rescheduling/plot_res/', "0930_statistics.csv")
    summary_df.to_csv(summary_save_path)
    print(f"Saved summary statistics to {summary_save_path}")


    print("\nSummary Statistics:")
    print(summary_df)

    print("All plots and summary statistics generated.")