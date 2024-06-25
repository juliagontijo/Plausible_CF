from collections import defaultdict
from itertools import cycle
import os
import re
import matplotlib.pyplot as plt
import numpy as np

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

def get_difference(dens_first, dens_second):
    return [key for key in dens_first if key not in dens_second]

def number_exists_in_dict(dens, instance):
    return instance in dens

def find_instances_with_smallest_mean_density(densities, first_param, second_param, top_x):
    instance_means = {}

    for instance, params in densities.items():
        filtered_densities = []
        for param, density in params.items():
            parts = param.split('-')
            if float(parts[0]) == first_param and float(parts[1]) == second_param:
                filtered_densities.append(density)
        
        if filtered_densities:
            instance_means[instance] = np.mean(filtered_densities)

    if not instance_means:
        return []

    sorted_instances = sorted(instance_means.items(), key=lambda item: item[1])[:top_x]
    return sorted_instances

def find_instances_with_largest_mean_density(densities, first_param, second_param, top_x):
    instance_means = {}

    for instance, params in densities.items():
        filtered_densities = []
        for param, density in params.items():
            parts = param.split('-')
            if float(parts[0]) == first_param and float(parts[1]) == second_param:
                filtered_densities.append(density)
        
        if filtered_densities:
            instance_means[instance] = np.mean(filtered_densities)

    if not instance_means:
        return []

    sorted_instances = sorted(instance_means.items(), key=lambda item: item[1], reverse=True)[:top_x]
    return sorted_instances

def main():
    base_folder = "/code/output_sum"  # Replace with your base folder path
    densities_sum = extract_densities(base_folder)

    for key, value in densities_sum.items():
        print(key)

    # base_folder = "/Users/juliagontijolopes/Desktop/GENERATED_GERMAN_50/output_sub"  # Replace with your base folder path
    # densities_sub = extract_densities(base_folder)

    # print(get_difference(densities_sum, densities_sub)) # what is in sum that is not in sub
    # print(get_difference(densities_sub, densities_sum)) # what is in sub that is not in sum
    # one_one = find_instances_with_smallest_mean_density(densities_sum, 1, 1, 50)
    # zero_zero = find_instances_with_smallest_mean_density(densities_sum, 0.2, 0.2, 50)
    # print("Smallest on 1 1:\n" + str(one_one))
    # print("Smallest on 0.2 0.2:\n" + str(zero_zero))

    # smallest_instance_numbers = set(instance for instance, _ in one_one)
    # largest_instance_numbers = set(instance for instance, _ in zero_zero)

    # common_instances = smallest_instance_numbers.intersection(largest_instance_numbers)
    # print("\n\nCOMMON: \n" + str(common_instances))
    

    # key_to_check = input("Enter the key to check: ")
    # while(key_to_check != "0"):
    #     print(number_exists_in_dict(densities_sum, key_to_check))
    #     key_to_check = input("Enter the key to check: ")
    
    print("\ndone!")
if __name__ == "__main__":
    main()
