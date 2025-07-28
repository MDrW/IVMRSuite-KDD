import matplotlib.pyplot as plt
import os
import numpy as np


def plot_train_results(result_history, figure_save_path):
    epochs = range(len(result_history['train_item_loss']))
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(2, 2, 1)
    ax.set_title('Train and Valid Item Loss')
    plt.plot(epochs, result_history['train_item_loss'], 'blue', label='Train Item Loss')
    plt.plot(epochs, np.array(result_history['valid_item_loss']).mean(axis=-1), 'green', label='Valid Item Loss')
    plt.legend()

    ax = plt.subplot(2, 2, 2)
    ax.set_title('Train and Valid Box Loss')
    plt.plot(epochs, result_history['train_box_loss'], 'blue', label='Train Box Loss')
    plt.plot(epochs, np.array(result_history['valid_box_loss']).mean(axis=-1), 'green', label='Valid Box Loss')
    plt.legend()
    
    ax = plt.subplot(2, 2, 3)
    ax.set_title('Train and Valid Item Acc')
    plt.plot(epochs, result_history['train_item_acc'], 'blue', label='Train Item Acc')
    plt.plot(epochs, np.array(result_history['valid_item_acc']).mean(axis=-1), 'green', label='Valid Item Acc(Mean)')
    plt.plot(epochs, np.array(result_history['valid_item_acc']).min(axis=-1), 'palegreen', label='Valid Item Acc(Min)')
    plt.plot(epochs, np.array(result_history['valid_item_acc']).max(axis=-1), 'lightgreen', label='Valid Item Acc(Max)')
    plt.legend()

    ax = plt.subplot(2, 2, 4)
    ax.set_title('Train and Valid Box Acc')
    plt.plot(epochs, result_history['train_box_acc'], 'blue', label='Train Box Acc')
    plt.plot(epochs, np.array(result_history['valid_box_acc']).mean(axis=-1), 'green', label='Valid Box Acc(Mean)')
    plt.plot(epochs, np.array(result_history['valid_box_acc']).min(axis=-1), 'palegreen', label='Valid Box Acc(Min)')
    plt.plot(epochs, np.array(result_history['valid_box_acc']).max(axis=-1), 'lightgreen', label='Valid Box Acc(Max)')
    plt.legend()
    plt.savefig(figure_save_path+'_loss_acc.jpg')
    
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 3, 1)
    ax.set_title('Train and Valid Acc')
    plt.plot(epochs, result_history['train_acc'], 'blue', label='Train Acc')
    plt.plot(epochs, np.array(result_history['valid_acc']).mean(axis=-1), 'green', label='Valid Acc(Mean)')
    plt.plot(epochs, np.array(result_history['valid_acc']).min(axis=-1), 'palegreen', label='Valid Acc(Min)')
    plt.plot(epochs, np.array(result_history['valid_acc']).max(axis=-1), 'lightgreen', label='Valid Acc(Max)')
    plt.legend()

    ax = plt.subplot(1, 3, 2)
    ax.set_title('Train and Valid Final Score')
    plt.plot(epochs, np.array(result_history['train_final_score']).mean(axis=-1), 'blue', label='Train Final Score(Mean)')
    plt.plot(epochs, np.array(result_history['train_final_score']).min(axis=-1), 'powderblue', label='Train Final Score(Min)')
    plt.plot(epochs, np.array(result_history['train_final_score']).max(axis=-1), 'lightblue', label='Train Final Score(Max)')
    plt.plot(epochs, np.array(result_history['valid_final_score']).mean(axis=-1), 'green', label='Valid Final Score(Mean)')
    plt.plot(epochs, np.array(result_history['valid_final_score']).min(axis=-1), 'palegreen', label='Valid Final Score(Min)')
    plt.plot(epochs, np.array(result_history['valid_final_score']).max(axis=-1), 'lightgreen', label='Valid Final Score(Max)')
    plt.legend()
    
    ax = plt.subplot(1, 3, 3)
    ax.set_title('Train and Valid Reward')
    plt.plot(epochs, np.array(result_history['train_reward']).mean(axis=-1), 'blue', label='Train Reward(Mean)')
    plt.plot(epochs, np.array(result_history['train_reward']).min(axis=-1), 'powderblue', label='Train Reward(Min)')
    plt.plot(epochs, np.array(result_history['train_reward']).max(axis=-1), 'lightblue', label='Train Reward(Max)')
    plt.plot(epochs, np.array(result_history['valid_reward']).mean(axis=-1), 'green', label='Valid Reward(Mean)')
    plt.plot(epochs, np.array(result_history['valid_reward']).min(axis=-1), 'palegreen', label='Valid Reward(Min)')
    plt.plot(epochs, np.array(result_history['valid_reward']).max(axis=-1), 'lightgreen', label='Valid Reward(Max)')
    plt.legend()
    plt.savefig(figure_save_path+'_reward.jpg')


if __name__ == '__main__':
    result_path = '../results/ecs_imitation/results/result_history/ecs_nocons_finite_easy_gnn_v1(t0_0.02).npy'
    result_history = np.load(result_path, allow_pickle=True)
    figure_save_path = '../results/ecs_imitation/results/result_figures/ecs_nocons_finite_easy_gnn_v1(t0_0.02)'
    plot_train_results(result_history=result_history.item(), figure_save_path=figure_save_path)
