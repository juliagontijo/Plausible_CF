#German
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import seaborn as sn

from csse import CSSE
from prepare_dataset import *

def main():
    # Read Dataset German
    df = prepare_KC2_dataset("KC2.csv", "/data/")

    #Get the input features
    columns = df.columns
    class_name = 'defects' # defects = 0 = "Good class" / defects = 1 = "Bad class" 
    columns_tmp = list(columns)
    columns_tmp.remove(class_name)

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(df[columns_tmp], df[class_name], range(len(df)),test_size=0.1)

    model = RandomForestClassifier(n_estimators = 120, n_jobs=-1, random_state=0)  
    model.fit(x_train, y_train)

    p = model.predict(x_test)

    print(classification_report(y_test, p))

    #-------Begin Parameter Adjustment--------
       
    X = 0 #Indicates the instance's position to be explained in the dataset

    #User preferences
    #static_list = [] #List of features that cannot be changed. For example: static_list = ['age']
    K = 5 #Number of counterfactual explanations to be obtained

    #Genetic Algorithm parameters
    #num_gen = 30 #number of generations
    #pop_size = 150 #population size
    #per_elit = 0.1 #percentage of elitism
    #cros_proba = 0.8 #crossover probability
    #mutation_proba = 0.1 #mutation probability

    # #Density Model parameters
    #distance_threshold = 1
    #density_threshold = np.logspace(-5, -2, 5)[0]
    #howmanypaths=10

    #Weights of objective function metrics
    L1 = 0.2 #lambda 1 - Weight assigned the distance to the original instance
    L2 = 0.2 #lambda 2 - Weight assigned the amount of changes needed in the original instance
    L3 = 1 #lambda 3 - Weight assigned the density to original instance

    #copy the original instance
    original_instance = x_test.iloc[X].copy()
    original_instance_index = idx_test[0] # corresponding index of original instance on original df
       
    #-------End Parameter Adjustment--------

    folder = "output/"
    output_file_name = folder + "output.txt"
    f = open(output_file_name, "w")
    f.write("L1 = " + str(L1) + "  L2 = " + str(L2) + "  L3 = " + str(L3) + "\n")

    print('Original instance - Class ' + str(p[X]) + '\n')
    print(original_instance)
    print("Original Instance row number = " + str(original_instance_index))
    print('\nGetting counterfactuals...\n')

    f.write('\n -- Original instance - Class ' + str(p[X]) + ' -- \n')
    f.write(str(original_instance))

    f.close()

    #Run CSSE - Method executed with default parameters except for the value of K.
    explainerCSSE = CSSE(df[columns_tmp], df[class_name], model, K = K, L1 = L1, L2 = L2, L3 = L3)
    
    contrafactual_set, solution = explainerCSSE.explain(original_instance, p[X], output_file_name, original_instance_index) #Method returns the list of counterfactuals and the explanations generated from them
    # np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    #The method returns a list of counterfactual solutions, where each solution, in turn, is a change list (each change has the "column" and "value" to be changed). To implement another output format, see the "printResults" function
    explainerCSSE.printResults(solution)

    explainerCSSE.printResultsOutputFile(solution, output_file_name)


    # ##### PCA PLOT BEGIN - when need to reduce density only
    # pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    # pca_result = pca.fit_transform(df[columns_tmp])
    # pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])

    # contrafactual_pca = pca.transform(contrafactual_set)
    # original_instance_pca = pca.transform([original_instance])

    # # Plot KDE based on the transformed df[columns_tmp]
    # plt.figure(figsize=(10, 7))
    # sn.kdeplot(data=pca_df, x='PC1', y='PC2', fill=True, cmap="Blues", alpha=0.5)

    # # Plot contrafactuals and the original instance
    # sn.scatterplot(x=contrafactual_pca[:, 0], y=contrafactual_pca[:, 1], color='red', marker='o', label='Contrafactuals')
    # sn.scatterplot(x=original_instance_pca[:, 0], y=original_instance_pca[:, 1], color='green', marker='X', label='Original Instance')

    # plt.legend()
    # plt.show()
    # ##### PCA PLOT END




    #  ##### MULTIPLE GRID PLOT BEGIN
    # for i, feature1 in enumerate(columns_tmp):
    #     for feature2 in columns_tmp[i+1:]:
    #         plt.figure(figsize=(6, 4))
    #         sn.kdeplot(data=df, x=feature1, y=feature2, fill=True, cmap="Blues", alpha=0.5)
    #         sn.scatterplot(x=contrafactual_set[feature1], y=contrafactual_set[feature2], color='red', marker='o', label='Contrafactuals')
    #         sn.scatterplot(x=[original_instance[feature1]], y=[original_instance[feature2]], color='green', marker='X', label='Original Instance')
    #         plt.title(f'{feature1} vs {feature2}')
    #         plt.legend()
    #         plt.show()
    #  ##### MULTIPLE GRID PLOT END





    print("done!")

    #explainerCSSE.plot(x_train, y_train, contrafactual_set, original_instance)

if __name__ == "__main__":
    main()





# if MOCKED_INDEX in idx_train:
#         # If the instance is in x_train, get its position
#         train_position = idx_train.index(MOCKED_INDEX)
        
#         # Get the instance from x_train and y_train
#         instance_x = x_train.iloc[train_position]
#         instance_y = y_train.iloc[train_position]
        
#         # Remove the instance from x_train and y_train
#         x_train = x_train.drop(x_train.index[train_position])
#         y_train = y_train.drop(y_train.index[train_position])
        
#         # Add the instance to the first position of x_test and y_test
#         x_test = pd.concat([pd.DataFrame([instance_x], columns=x_test.columns), x_test], ignore_index=True)
#         y_test = pd.concat([pd.Series([instance_y]), y_test.reset_index(drop=True)], ignore_index=True)
        
#     elif MOCKED_INDEX in idx_test:
#         # If the instance is in x_test, get its position
#         test_position = idx_test.index(MOCKED_INDEX)
        
#         # Get the instance from x_test and y_test
#         instance_x = x_test.iloc[test_position]
#         instance_y = y_test.iloc[test_position]
        
#         # Remove the instance from x_test and y_test
#         x_test = x_test.drop(x_test.index[test_position])
#         y_test = y_test.drop(y_test.index[test_position])
        
#         # Add the instance to the first position of x_test and y_test
#         x_test = pd.concat([pd.DataFrame([instance_x], columns=x_test.columns), x_test], ignore_index=True)
#         y_test = pd.concat([pd.Series([instance_y]), y_test.reset_index(drop=True)], ignore_index=True)