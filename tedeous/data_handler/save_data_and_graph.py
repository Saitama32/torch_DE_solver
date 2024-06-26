import matplotlib.pyplot as plt
import os
import json

def save_data_and_graph(dict_of_learning, eq, name):

    plt.clf()

    folder_path = f"tedeous/data_handler/{eq}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    min_loss = min(dict_of_learning["learning_f_p_losses"][-1])
    num_particles = len(dict_of_learning["learning_f_p_losses"][0])  # количество частиц
    num_epoch = len(dict_of_learning["learning_f_p_losses"])
    error = dict_of_learning['RMSE_pso']
    for particle_idx in range(num_particles):
        losses = [iteration[particle_idx] for iteration in dict_of_learning["learning_f_p_losses"]]  # функции потерь для текущей частицы
        plt.plot(losses, label=f'Particle {particle_idx}')

    plt.ylim(0.0, 1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Loss curves for {num_particles} particles {name}')
    plt.axhline(y=min_loss, color='b', linestyle='--', label=f'Min Loss: {min_loss}')
    plt.axhline(y=error, color='g', linestyle='--', label=f'RMSE_pso: {error}')
    plt.text(0, min_loss, f'Min Loss: {min_loss:.8f}', color='blue', ha='left', va='bottom')
    plt.text(num_epoch, error, f'RMSE_pso: {error:.8f}', color='green', ha='right', va='bottom')
    file_path = f'tedeous/data_handler/{eq}/{error:.8f}_{min_loss:.8f}.png'
    plt.savefig(file_path, format='png', dpi=300)
    file_path = f'tedeous/data_handler/{eq}/{error:.8f}_{min_loss:.8f}'
    with open(file_path, 'w') as f:
        json.dump(dict_of_learning, f)