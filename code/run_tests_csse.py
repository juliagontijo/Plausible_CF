import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import seaborn as sn

from csse import CSSE
from prepare_dataset import *

def run_test(MOCKED_INDEX, L1, L2, L3, df, columns_tmp, class_name):
    original_instance = df.iloc[MOCKED_INDEX].copy()

    mask = df.index != MOCKED_INDEX

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        df[mask][columns_tmp], df[mask][class_name], range(len(df[mask])),
        test_size=0.1, random_state=42
    )

    idx_train = list(idx_train)
    idx_test = list(idx_test)

    x_test = pd.concat([pd.DataFrame([original_instance[columns_tmp]]), x_test], ignore_index=True)
    y_test = pd.concat([pd.Series([original_instance[class_name]]), y_test.reset_index(drop=True)], ignore_index=True)

    idx_test = [MOCKED_INDEX] + [i for i in idx_test]

    model = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=0)
    model.fit(x_train, y_train)

    p = model.predict(x_test)

    print(classification_report(y_test, p))

    X = 0  # Indicates the instance's position to be explained in the dataset
    original_instance = x_test.iloc[X].copy()
    original_instance_index = MOCKED_INDEX

    instance_folder = os.path.join("output_sum", str(MOCKED_INDEX))
    if not os.path.exists(instance_folder):
        os.makedirs(instance_folder)

    config_folder = os.path.join(instance_folder, f"{L1}-{L2}-{L3}")
    if not os.path.exists(config_folder):
        os.makedirs(config_folder)

    test_folder = os.path.join(config_folder, "Teste-1")
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    output_file_name = os.path.join(test_folder, "output.txt")
    with open(output_file_name, "w") as f:
        f.write(f"L1 = {L1}  L2 = {L2}  L3 = {L3}\n")

        f.write('\n -- Original instance - Class ' + str(p[X]) + ' -- \n')
        f.write(str(original_instance))
        f.write("\nOriginal Instance row number = " + str(original_instance_index) + "\n")

    f.close()

    explainerCSSE = CSSE(df[columns_tmp], df[class_name], model, K=5, L1=L1, L2=L2, L3=L3)
    contrafactual_set, solution = explainerCSSE.explain(original_instance, p[X], output_file_name, test_folder, original_instance_index)
    # explainerCSSE.printResults(solution)
    explainerCSSE.printResultsOutputFile(solution, output_file_name)

    print("Test completed for MOCKED_INDEX:", MOCKED_INDEX, "with L3 =", L3)

def main():
    df = prepare_breast_winsconsin_dataset("breast_winsconsin.csv", "/data/")
    columns = df.columns
    class_name = 'diagnosis'
    columns_tmp = list(columns)
    columns_tmp.remove(class_name)

    L_values = [1, 0.2]
    L3_values = [0, 0.2, 0.4, 0.6, 0.8, 1]

    with open("instances_lists/instances_list_breast_winsconsin.txt", "r") as file:
        instances = [int(line.strip()) for line in file]

    for MOCKED_INDEX in instances:
        for L in L_values:
            for L3 in L3_values:
                run_test(MOCKED_INDEX, L, L, L3, df, columns_tmp, class_name)

if __name__ == "__main__":
    main()
