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

from csse import density
from prepare_dataset import *

def main():
    df = prepare_german_dataset("generated_german_credit_with_cf.csv", "/data/")
    MOCKED_INDEX = 379
    original_instance = df.iloc[MOCKED_INDEX].copy()

    #Get the input features
    columns = df.columns
    class_name = 'default' # default = 0 = "Good class" / default = 1 = "Bad class" 
    columns_tmp = list(columns)
    columns_tmp.remove(class_name)
    mask = df.index != MOCKED_INDEX

    # x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(df[columns_tmp], df[class_name], range(len(df)),test_size=0.1)
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
    df[mask][columns_tmp], df[mask][class_name], range(len(df[mask])),
    test_size=0.1, random_state=42)

    idx_train = list(idx_train)
    idx_test = list(idx_test)

    x_test = pd.concat([pd.DataFrame([original_instance[columns_tmp]]), x_test], ignore_index=True)
    y_test = pd.concat([pd.Series([original_instance[class_name]]), y_test.reset_index(drop=True)], ignore_index=True)

    # Update the test indices to reflect the original instance
    idx_test = [MOCKED_INDEX] + [i for i in idx_test]


    model = RandomForestClassifier(n_estimators = 120, n_jobs=-1, random_state=0)  
    model.fit(x_train, y_train)

    p = model.predict(x_test)

    print(classification_report(y_test, p))

    K = 5

    f = open("just_path_output", "w")
    density_computation = density(model)
    density_computation.generateSpaceDensity(df[columns_tmp], df[class_name], f)

    density_computation.getGraph()
    paths, recom = density_computation.compute_path(MOCKED_INDEX, list(num for num in range((df[columns_tmp].shape[0] - K), df[columns_tmp].shape[0])), f)
    density_computation.print_recommendation(paths, recom, f)

    f.close()

if __name__ == "__main__":
    main()
