from collections import defaultdict
from itertools import cycle
import os
import re
import matplotlib.pyplot as plt
import numpy as np

sumsub = "sum"

def extract_densities(base_folder):
    densities = {}

    for instance_folder in os.listdir(base_folder):
        instance_path = os.path.join(base_folder, instance_folder)
        if os.path.isdir(instance_path):
            densities[instance_folder] = {}

            for param_folder in os.listdir(instance_path):
                param_path = os.path.join(instance_path, param_folder)
                if os.path.isdir(param_path):
                    test_folder = os.path.join(param_path, "Teste-1")
                    if os.path.isdir(test_folder):
                        output_file = os.path.join(test_folder, "output.txt")
                        if os.path.exists(output_file):
                            with open(output_file, "r") as f:
                                content = f.read()
                                match = re.search(r"Average density: ([\d.e+-]+)", content)
                                if match:
                                    density = float(match.group(1))
                                    densities[instance_folder][param_folder] = density

    return densities

def plot_densities_by_instance(densities, instance):
    if instance not in densities:
        print(f"Instance {instance} not found.")
        return

    params = []
    values = []

     # Extract and sort the hyperparameters
    sorted_items = sorted(densities[instance].items(), key=lambda x: tuple(map(float, x[0].split('-'))))

    for param, density in sorted_items:
        params.append(param)
        values.append(density)

    plt.figure(figsize=(10, 6))
    plt.bar(params, values, color='blue')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Average Density')
    plt.title(f'Average Densities for Instance {instance}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = os.path.join("plots_per_instance_" + sumsub, f"{instance}")
    plt.savefig(plot_path)
    plt.close()

def plot_densities_by_hyperparameters(densities, hyperparameters):
    instances = []
    values = []

    for instance, params in densities.items():
        if hyperparameters in params:
            instances.append(instance)
            values.append(params[hyperparameters])

    plt.figure(figsize=(10, 6))
    plt.bar(instances, values, color='green')
    plt.xlabel('Instance')
    plt.ylabel('Average Density')
    plt.title(f'Average Densities for Hyperparameters {hyperparameters}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    # plot_path = os.path.join("plots", f"{}")
    # plt.savefig(plot_path)
    # plt.close()
    plt.show()

def plot_all_densities_by_instances(densities):
    instances = list(densities.keys())
    all_params = sorted({param for instance in instances for param in densities[instance].keys()}, key=lambda x: tuple(map(float, x.split('-'))))
    
    param_colors = {}
    color_cycle = cycle(plt.cm.tab20.colors)  # Use a colormap with 20 different colors

    for param in all_params:
        param_colors[param] = next(color_cycle)

    bar_width = 0.8 / len(instances)  # Width of each bar group

    fig, ax = plt.subplots(figsize=(15, 8))
    bar_positions = np.arange(len(instances)) * (len(all_params) + 1)  # Add space between instance groups

    for param in all_params:
        values = [densities[instance].get(param, 0) for instance in instances]
        ax.bar(bar_positions, values, bar_width, label=param, color=param_colors[param])
        bar_positions += bar_width

    ax.set_xlabel('Instances')
    ax.set_ylabel('Average Density')
    ax.set_title('Average Densities for All Instances and Hyperparameters')
    ax.set_xticks(np.arange(len(instances)) * (len(all_params) + 1) + (len(all_params) - 1) / 2 * bar_width)
    ax.set_xticklabels(instances, rotation=90)
    ax.legend(title='Hyperparameters')
    plt.tight_layout()
    plt.show()
    
def plot_mean_densities_for_group_bar(densities, first_param, second_param):
    mean_densities = defaultdict(list)
    
    for instance, params in densities.items():
        for param, density in params.items():
            parts = param.split('-')
            if float(parts[0]) == first_param and float(parts[1]) == second_param:
                mean_densities[param].append(density)
    
    sorted_params = sorted(mean_densities.keys(), key=lambda x: float(x.split('-')[2]))
    means = [np.mean(mean_densities[param]) for param in sorted_params]

    plt.figure(figsize=(10, 6))
    plt.bar(sorted_params, means, color='purple')
    plt.xlabel('Hyperparameters')
    plt.ylabel('Mean Average Density')
    plt.title(f'Mean Average Densities for Hyperparameters starting with {first_param}-{second_param}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plot_path = os.path.join("plots_mean_dens_per_L1L2_" + sumsub, f"{first_param}-{second_param}.png")
    plt.savefig(plot_path)
    plt.close()

def plot_mean_densities_for_group_point(densities, first_param, second_param):
    mean_densities = defaultdict(list)
    
    for instance, params in densities.items():
        for param, density in params.items():
            parts = param.split('-')
            if float(parts[0]) == first_param and float(parts[1]) == second_param:
                mean_densities[param].append(density)
    
    sorted_params = sorted(mean_densities.keys(), key=lambda x: float(x.split('-')[2]))
    means = [np.mean(mean_densities[param]) for param in sorted_params]
    variances = [np.var(mean_densities[param]) for param in sorted_params]
    
    x_values = [float(param.split('-')[2]) for param in sorted_params]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, means, marker='o', color='purple', label='Densidade média')
    for i, mean in enumerate(means):
        plt.text(x_values[i], mean, f"{mean:.2e}", fontsize=9, ha='center', va='bottom')
    plt.xlabel('Variação de L 3')
    plt.ylabel('Densidade média')
    plt.title(f'Densidade média para L {first_param} - L {second_param}')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join("plots_mean_dens_per_L1L2_" + sumsub, f"{first_param}-{second_param}.png")
    plt.savefig(plot_path)
    plt.close()

def main():
    base_folder = "/Users/juliagontijolopes/Desktop/TRYING_DENSITY/code/output_" + sumsub  # Replace with your base folder path
    # base_folder = "/Users/juliagontijolopes/Desktop/VERTEBRAL_COLUMN_50/output_" + sumsub
    densities = extract_densities(base_folder)

    # Example usage:
    # # Plot densities for a specific instance
    # for instance, params in densities.items():
    #     plot_densities_by_instance(densities, instance) 

    plot_mean_densities_for_group_point(densities, 1, 1)
    plot_mean_densities_for_group_point(densities, 0.2, 0.2)
    # Plot densities for specific hyperparameters across all instances
    # plot_densities_by_hyperparameters(densities, "1-1-0.6")  # Replace "1-1-0.6" with your hyperparameters

    # Plot all densities by instances
    # plot_all_densities_by_instances(densities)
    
    print("done!")
if __name__ == "__main__":
    main()
