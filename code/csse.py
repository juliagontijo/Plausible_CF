import os
import random as rnd
from matplotlib import ticker
from scipy.spatial import distance
import pandas as pd
from sklearn import datasets, preprocessing, manifold
import warnings
from tqdm import tqdm
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

from dijsktra_algorithm import Graph, dijsktra_tosome
# from Orange.projection import FreeViz

import warnings

# To ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# folder = "output/"

# testWeights = "1-1-1-"
# testWeights = "1-1-0.2-"
# testWeights = "0.2-0.2-1-"

# testDensityType = "combined/"
# testDensityType = "just-point/"
# testDensityType = "/"

# path_to_image = folder

#Used for ordering evaluations
class individual:
    def __init__(self, index, score, distance, num_changes, aval_norm, dist_norm, predict_proba, density):
        self.index = index #Indicates the instance's position in the dataframe
        self.score = score #Indicates the score in relation to the proximity of the class boundary
        self.distance = distance #Indicates the distance from the original instance
        self.num_changes = num_changes #Indicates the number of changes for class change
        self.aval_norm = aval_norm #Indicates the final fitness with standardized metrics
        self.dist_norm = dist_norm #Indicates the normalized distance (distance and number of changes)
        self.predict_proba = predict_proba #Indicates de individual's class
        self.density = density
    def __repr__(self):
        return repr((self.index, self.score, self.distance, self.num_changes, self.aval_norm, self.dist_norm, self.predict_proba, self.density))

class counter_change:
    def __init__(self, column, value):
        self.column = column 
        self.value = value
    def __eq__(self, other):
        if self.column == other.column and self.value == other.value:
            return True
        else:
            return False
    def __repr__(self):
        return repr((self.column, self.value))    

#Used to generate a random value in the mutation operation
class feature_range:
    def __init__(self, column, col_type, min_value, max_value):
        self.column = column 
        self.col_type = col_type
        self.min_value = min_value
        self.max_value = max_value

    #Returns a random value to perform mutation operation
    def get_random_value(self):
        if self.col_type == 'int64' or self.col_type == 'int' or self.col_type == 'int16' or self.col_type == 'int8' or (self.col_type == 'uint8'):
            value = rnd.randint(self.min_value, self.max_value)
        else:  
            value = round(rnd.uniform(self.min_value, self.max_value), 2)
        return value
    
    #Checks if the attribute has only one value.
    def unique_value(self):
        if self.min_value != self.max_value:
            return False
        else:  
            return True    

    def __repr__(self):
        return repr((self.column, self.col_type, self.min_value, self.max_value)) 
        
class CSSE(object):
    
    def __init__(self, input_dataset, input_dataset_class, model, static_list = [], K = 5, num_gen = 30, pop_size = 100, per_elit = 0.1, cros_proba = 0.8, mutation_proba = 0.1, L1 = 1, L2 = 1, L3 = 1):
        #User Options
        self.static_list = static_list #List of static features
        self.K = K #Number of counterfactuals desired
        #Model
        self.input_dataset = input_dataset
        self.input_dataset_class = input_dataset_class
        self.model = model
        #GA Parameters
        self.num_gen = num_gen
        self.pop_size = pop_size
        self.per_elit = per_elit
        self.cros_proba = cros_proba
        self.mutation_proba = mutation_proba
        #Objective function parameters
        self.L1 = L1 #weight assigned the distance to the original instance
        self.L2 = L2 #weight assigned the number of changes needed in the original instance
        self.L3 = L3 #weight assigned the density to the original instance
    
    #Get which index in the SHAP corresponding to the current class
    def getBadClass(self):   
        if self.current_class == self.model.classes_[0]:
            ind_cur_class = 0
        else:
            ind_cur_class = 1
        
        return ind_cur_class
    
    #Gets the valid values range for each feature
    def getFeaturesRange(self):
        features_range = []
       
        for i in range (0, self.input_dataset.columns.size):
            col_name = self.input_dataset.columns[i]
            col_type = self.input_dataset[col_name].dtype
            min_value = min(self.input_dataset[col_name])
            max_value = max(self.input_dataset[col_name])
            
            feature_range_ind = feature_range(col_name, col_type, min_value, max_value)
            features_range.append(feature_range_ind)
        
        return features_range
       
    def getMutationValue(self, currentValue, index, ind_feature_range):
        new_value = ind_feature_range.get_random_value()
        
        while currentValue == new_value:
            new_value = ind_feature_range.get_random_value()
        
        return new_value
    
    def equal(self, individual, population):
        aux = 0
        for i in range ( 1, len(population)):
            c = population.loc[i].copy()
            dst = distance.euclidean(individual, c)
            if dst == 0:
                aux = 1
        
        return aux

    def getPopInicial (self, df, features_range): 
        #The reference individual will always be in the 0 position of the df - so that it is normalized as well (it will be used later in the distance function)
        df.loc[0] = self.original_ind.copy()
        
        #Counting numbers of repeated individuals
        number_repetitions = 0
        
        #One more position is used because the zero position was reserved for the reference individual
        while len(df) < self.pop_size + 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = self.original_ind.copy()

                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[len(df)] = mutant.copy()
                else:
                    #Assesses whether the GA is producing too many repeated individuals.
                    number_repetitions = number_repetitions + 1
                    if number_repetitions == 2*self.pop_size:
                        self.pop_size = round(self.pop_size - self.pop_size*0.1)
                        self.mutation_proba = self.mutation_proba + 0.1
                        #print('Adjusting population size...', self.pop_size)
                        number_repetitions = 0
    
    #Complete the standardized proximity and similarity assessments for each individual
    def getNormalEvaluation(self, evaluation, aval_norma, f):
        # print("aval_norma", aval_norma)
        scaler2 = preprocessing.MinMaxScaler()
        aval_norma2 = scaler2.fit_transform(aval_norma)
    
        i = 0
        # f.write("\n\n______ Normal Evaluation ______ \n")
        while i < len(evaluation):
            evaluation[i].aval_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + aval_norma2[i,2] + self.L3*aval_norma2[i,3]
            evaluation[i].dist_norm = self.L1*aval_norma2[i,0] + self.L2*aval_norma2[i,1] + self.L3*aval_norma2[i,3]
        
            i = i + 1
            # f.write("\nIndividual " + str(i))
            # f.write("  -  Aval Norm = " + str(evaluation[i-1].aval_norm) + " Dist Norm = " + str(evaluation[i-1].dist_norm))
        
        # f.write("\n")
    
    def numChanges(self, ind_con):
        num = 0
        for i in range(len(self.original_ind)):
            if self.original_ind[i] != ind_con[i]:
                num = num + 1
        
        return num
        
    def fitness(self, population, evaluation, ind_cur_class, density_computation, f):
        def getProximityEvaluation (proba):
            #Penalizes the individual who is in the negative class
            if proba < 0.5:
                predict_score = 0
            else:
                predict_score= proba
             
            return predict_score
               
        #Calculates similarity to the original instance
        def getEvaluationDist (ind, X_train_minmax):
            #Normalizes the data so that the different scales do not bias the distance
            a = X_train_minmax[0]
            b = X_train_minmax[ind]
            dst = distance.euclidean(a, b)
  
            return dst

        #Calculates density on midpoint to the original instance
        def getDensityEvaluation (ind, X_train_minmax, density_computation, alpha = 0.5):
            # density1 = density_computation.get_kde_density_2points(X_train_minmax[ind], X_train_minmax[0], 1)
            density2 = density_computation.get_kde_density_point(X_train_minmax[ind], 1)
            # combined_density =(alpha * density1) + ((1 - alpha) * density2)
            # print("Density1 = " + str(density1) + " | Density2 = " + str(density2) + " | Combined = " + str(combined_density))
  
            return density2
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        
            predict_proba = self.model.predict_proba(population)
                    
        #Calculating the distance between instances
        scaler = preprocessing.MinMaxScaler()
        X_train_minmax = scaler.fit_transform(population)
    
        i = 0
        aval_norma = [] 
        # f.write("\n\n################ FITNESS ################ \n\n")
        # parentDens = density_computation.get_kde_density_point(X_train_minmax[0], 1)
        # f.write("\nParent of Pop Density = " + str(parentDens))
        # print("\n\nFITNESS\n\n")
        
        while i < len(population):
            proximityEvaluation = getProximityEvaluation(predict_proba[i, ind_cur_class])

            # f.write("\nIndividual " + str(i))
            # print("\n\n\nDens Parent Of Counterfactual Pop = " + str(parentDens))
            evaldist = getEvaluationDist(i, X_train_minmax)
            #The original individual is in the 1st position
            evaldensity = getDensityEvaluation(i, X_train_minmax, density_computation)
            numChanges = self.numChanges(population.loc[i])
            # f.write("  -  Eval Distance = " + str(evaldist) + " Eval NumChanges = " + str(numChanges) + " Eval Density = " + str(evaldensity))

            ind = individual(i, proximityEvaluation, evaldist, numChanges, 0, 0, predict_proba[i, ind_cur_class], evaldensity)
            aval_norma.append([evaldist, numChanges, proximityEvaluation, evaldensity])
            evaluation.append(ind)
            i = i + 1

        self.getNormalEvaluation(evaluation, aval_norma, f)
       
    #Given a counterfactual solution returns the list of modified columns
    def getColumns(self, counter_solution):
        colums = []
        for j in range (0, len(counter_solution)):
            colums.append(counter_solution[j].column)
        
        return colums      
             
    #Checks if the new solution is contained in the solutions already found
    def contained_solution(self, original_instance, current_list, current_column_list, new_solution, new_column_solution):
        contained = False
        for i in range (0, len(current_list)):              
            if set(current_column_list[i]).issubset(new_column_solution):
                for j in range (0, len(current_list[i])):
                    pos = new_column_solution.index(current_list[i][j].column)
                    distancia_a = abs(original_instance[current_list[i][j].column] - current_list[i][j].value)
                    distancia_b = abs(original_instance[current_list[i][j].column] - new_solution[pos].value)
                    if distancia_b >= distancia_a:
                        contained = True

        return contained
      
    def elitism(self, evaluation, df, parents):
         
        num_elit = round(self.per_elit*self.pop_size)
        
        aval = []
        aval = evaluation.copy()
        aval.sort(key=lambda individual: individual.aval_norm)
        
        #contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        while i < len(aval) and numContraf <= num_elit + 1:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, parents)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution                    
                        df.loc[len(df)] = parents.iloc[aval[i].index].copy()                     
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                      
            i = i + 1
        return solution_list
    
    def roulette_wheel(self, evaluation):
        summation = 0
        #Performs roulette wheel to select parents who will undergo genetic operations
        for i in range (1, len(evaluation)): 
            summation = summation + 1/evaluation[i].aval_norm
    
        roulette = rnd.uniform( 0, summation )
    
        roulette_score = 1/evaluation[1].aval_norm
        i = 1
        while roulette_score < roulette:
            i += 1
            roulette_score += 1/evaluation[i].aval_norm
        
        return i
            
    def crossover (self, df, parents, evaluation, number_cross_repetitions):
        child = []
            
        corte = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
        index1 = self.roulette_wheel(evaluation)
        index2 = self.roulette_wheel(evaluation)
        
        ind_a = parents.iloc[index1].copy()
        ind_b = parents.iloc[index2].copy()
            
        crossover_op = rnd.random()
        if crossover_op <= self.cros_proba:
            child[ :corte ] = ind_a[ :corte ].copy()
            child[ corte: ] = ind_b[ corte: ].copy()
        else:
            child = ind_a.copy()
        
        ni = self.equal(child, df)
        if ni == 0:
            df.loc[len(df)] = child.copy()
        else:
            #Assesses whether the GA is producing too many repeated individuals.
            number_cross_repetitions = number_cross_repetitions + 1
            if number_cross_repetitions == self.pop_size:
                self.pop_size = round(self.pop_size - self.pop_size*0.1)
                self.mutation_proba = self.mutation_proba + 0.1
                #print('Adjusting population size...', self.pop_size)
                number_cross_repetitions = 0
        #    print('repeated')
        return number_cross_repetitions
                       
    def mutation (self, df, individual_pos, features_range):
        ni = 1
        #Does not allow repeated individual
        while ni == 1:
            #Draw a feature to change
            index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            while df.columns[index_a] in self.static_list:
                index_a = rnd.randint( 0, self.input_dataset.columns.size - 1 )
            
            if not features_range[index_a].unique_value():
                #Mutation
                mutant = df.iloc[individual_pos].copy()
            
                #Draw the value to be changed
                new_value =  self.getMutationValue(mutant.iloc[index_a], index_a, features_range[index_a])  
                mutant.iloc[index_a] = new_value

                ni = self.equal(mutant, df)
                if ni == 0:
                    df.loc[individual_pos] = mutant.copy()
                #else:
                #    print('repeated')
     
    def getChanges(self, ind, dfComp):
        changes = []
        
        for i in range (len(dfComp.iloc[ind])):
            if self.original_ind[i] != dfComp.loc[ind][i]:
                counter_change_ind = counter_change(dfComp.columns[i], dfComp.loc[ind][i])
                changes.append(counter_change_ind)

        return changes
    
    #Generates the solution from the final population
    def getContrafactual(self, df, aval, f):
        
        contrafactual_ind = pd.DataFrame(columns=self.input_dataset.columns)
        solution_list = []
        solution_colums_list = []
        
        i = 0
        numContraf = 0
        sumDist = 0
        sumChanges = 0
        sumDensity = 0
        f.write("\n\n\n ==================== CONTRAFACTUAL RESULTS ==================== \n\n")
        while i < len(aval) and numContraf < self.K:
            #Checks if the example belongs to the counterfactual class
            if aval[i].predict_proba < 0.5:
                ind_changes = []
                ind_colums_change = []
         
                #Gets counterfactual example change list
                ind_changes = self.getChanges(aval[i].index, df)
                #Generates the list of columns modified in the counterfactual to check if there is already a solution with that set of columns
                ind_colums_change = self.getColumns(ind_changes)
                
                if ind_colums_change not in solution_colums_list:
                    #Check if one solution is a subset of the other
                    if not self.contained_solution(self.original_ind, solution_list, solution_colums_list, ind_changes, ind_colums_change):
                        #Include counterfactual in the list of examples of the final solution
                        contrafactual_ind.loc[len(contrafactual_ind)] = df.iloc[aval[i].index].copy()
                                
                        #Add to the list of solutions (changes only)       
                        solution_list.append(ind_changes)
                        #Used to compare with the next counterfactuals (to ensure diversity)
                        solution_colums_list.append(ind_colums_change)
                                        
                        numContraf = numContraf + 1
                        sumDist = sumDist + aval[i].distance
                        sumChanges = sumChanges + aval[i].num_changes
                        sumDensity = sumDensity + aval[i].density
                        f.write("\nContrafactual " + str(numContraf) + "\nDistance = " + str(aval[i].distance) + " | NumChanges = " + str(aval[i].num_changes)+ " | Density = " + str(aval[i].density)) 
                        #print('solution_list ', solution_list)
                    #else:
                        #print('is contained ', ind_changes)
                #else:
                    #print('repeated ', ind_changes)
                      
            i = i + 1

        f.write("\n\nAverage of all counterfactuals " + "\nDistance = " + str(sumDist/numContraf) + " | NumChanges = " + str(sumChanges/numContraf) + " | Density = " + str(sumDensity/numContraf))

        return contrafactual_ind, solution_list   
    
    def printResults(self, solution):
        print("Result obtained")
        if len(solution) != 0:
            for i in range(0, len(solution)): 
                print("\n")
                print(f"{'Counterfactual ' + str(i + 1):^34}")
                for j in range(0, len(solution[i])): 
                    print(f"{str(solution[i][j].column):<29} {str(solution[i][j].value):>5}")
        else:
            print('Solution not found. It may be necessary to adjust the parameters for this instance.')

    def printResultsOutputFile(self, solution, outputfilename):
        f = open(outputfilename, "a")
        f.write("\n\n\n******** Result obtained ********\n")
        if len(solution) != 0:
            for i in range(0, len(solution)): 
                f.write("\n\n")
                f.write(f"{'Counterfactual ' + str(i + 1):^34}\n")
                for j in range(0, len(solution[i])): 
                    f.write(f"{str(solution[i][j].column):<29} {str(solution[i][j].value):>5}\n")
        else:
            f.write('\nSolution not found. It may be necessary to adjust the parameters for this instance.\n')

        f.close()
                                                 
    def explain(self, original_ind, current_class, output_file_name, test_folder, original_instance_index):
        self.original_ind = original_ind #Original instance
        #self.ind_cur_class = ind_cur_class #Index in the shap corresponds to the original instance class
        self.current_class = current_class #Original instance class
        
        ind_cur_class = self.getBadClass()
    
        #Gets the valid values range of each feature
        features_range = []
        features_range = self.getFeaturesRange()

        #The DataFrame df will have the current population
        df = pd.DataFrame(columns=self.input_dataset.columns)

        plot_TSNE(self.input_dataset, self.input_dataset_class, 50, False, self.K, original_instance_index, test_folder, 'BEFORE_CSSE.png')
        # plot_FreeViz(self.input_dataset, self.input_dataset_class, False, self.K)

        #Calculate dataset space density
        f = open(output_file_name, "a")
        density_computation = density(self.model)
        density_computation.generateSpaceDensity(self.input_dataset, self.input_dataset_class, f)
        # f.write("\n\n######## Raw Dataset Density BEGGIN ######## \n\n")
        # pop_density = density_computation.get_kde_density_population(self.input_dataset, 1, f)
        # min_density_index = pop_density.argmin()
        # min_density_value = pop_density[min_density_index]
        # min_density_instance = df.iloc[min_density_index]
        # f.write("Min density: " + str(min(pop_density)) + " | Row number: " + str(pop_density.index(min(pop_density))))
        # f.write("\nPop Density After Function\n")
        # f.write(str(pop_density))
        # f.write("\n\n######## Raw Dataset Density END ######## \n\n")
        
        
        #Generates the initial population with popinitial mutants        
        self.getPopInicial(df, features_range)

        # # PLOTTING INITIAL POP
        # # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
        # def pred_func(x):
        #         return self.model.predict_proba(x)[:, 1]
        
        # plot_decision_boundary_initial(X, y, pred_func)
        # # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL

        
        for g in tqdm(range(self.num_gen), desc= "Processing..."):
            #To use on the parents of each generation
            parents = pd.DataFrame(columns=self.input_dataset.columns)
    
            #Copy parents to the next generation
            parents = df.copy()
            
            #df will contain the new population
            df = pd.DataFrame(columns=self.input_dataset.columns)
            
            evaluation = []                         
                   
            #Assessing generation counterfactuals
            self.fitness(parents, evaluation, ind_cur_class, density_computation, f)
            
            #The original individual will always be in the 0 position of the df - So that it is normalized too (it will be used later in the distance function)
            df.loc[0] = self.original_ind.copy()
            
            #Copies to the next generation the per_elit best individuals
            self.elitism(evaluation, df, parents)
            
            number_cross_repetitions = 0
            while len(df) < self.pop_size + 1: #+1, as the 1st position is used to store the reference individual
                number_cross_repetitions = self.crossover(df, parents, evaluation, number_cross_repetitions)
                
                mutation_op = rnd.random()
                if mutation_op <= self.mutation_proba:
                    self.mutation(df, len(df) - 1, features_range)
            
            # PLOTTING EACH GENERATION
            # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
                    
            #Getting the counterfactual set
            # contrafactual_set_aux = pd.DataFrame(columns=self.input_dataset.columns)
            # contrafactual_set_aux, solution_list_aux = self.getContrafactual(df, evaluation) 

            # def pred_func(x):
            #         return self.model.predict_proba(x)[:, 1]
            
            #plot_decision_boundary(X, y, pred_func, contrafactual_set_aux)
            # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
                 

        evaluation = []
    
        #Evaluating the latest generation
        self.fitness(df, evaluation, ind_cur_class, density_computation, f)
    
        #Order the last generation by distance to the original instance     
        evaluation.sort(key=lambda individual: individual.aval_norm)     
        
        #Getting the counterfactual set

        contrafactual_set = pd.DataFrame(columns=self.input_dataset.columns)
        contrafactual_set, solution_list = self.getContrafactual(df, evaluation, f)
        popCF_density = density_computation.get_kde_density_population(contrafactual_set, 1, f)
        f.write("\n\nFinal Counterfactuals Point Density\n")
        f.write(str(popCF_density))
        density_values = [d[0] for d in popCF_density]
        average_density = sum(density_values) / len(density_values)
        f.write("\nAverage density: " + str(average_density))

        # # UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL
        # # def pred_func(x):
        # #         return self.model.predict_proba(x)[:, 1]
        
        # plot_decision_boundary(X, y, pred_func, contrafactual_set)
        # # STOP UNCOMMENT WHEN PLOTTING 2D DATASET COUNTERFACTUAL

        f.write("\n\n\n################ GRAPH BEGGIN ################ \n\n")

        x_withContrafactuals = np.concatenate([self.input_dataset, contrafactual_set])
        contrafactual_classes = self.model.predict(contrafactual_set)
        # contrafactual_classes = np.full(shape=(contrafactual_set.shape[0],), fill_value=1)
        # newy = pd.Series(np.zeros(len(contrafactual_set)))
        y_withContrafactuals = np.concatenate([self.input_dataset_class, contrafactual_classes])

        plot_TSNE(x_withContrafactuals, y_withContrafactuals, 50, True, self.K, original_instance_index, test_folder,'AFTER1_CSSE.png')
        plot_TSNE2(self.input_dataset, self.input_dataset_class, 50, contrafactual_set, original_instance_index, self.model, test_folder, 'AFTER2_CSSE.png')

        # density_computation.generateSpaceDensity(x_withContrafactuals, y_withContrafactuals, f)
        # f.close()
        # f = open(output_file_name, "a")
        # density_computation.getGraph()
        # paths, recom = density_computation.compute_path(original_instance_index, list(num for num in range((x_withContrafactuals.shape[0] - self.K), x_withContrafactuals.shape[0])), f)
        # density_computation.print_recommendation(paths, recom, f)

        f.close()
                 
        return contrafactual_set, solution_list



def plot_TSNE(X, y, perpl, hasContrafactual, k, original_instance_index, test_folder, path):
    X_scaled = normalize_if_needed(X)
    
    t_sne = manifold.TSNE(
        init = 'pca',
        n_components=2, #make it not hard coded
        perplexity=perpl,
        n_iter=2000,
        random_state=0
    )

    S_t_sne = t_sne.fit_transform(X_scaled)
    print(t_sne.kl_divergence_)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Categorical(y).codes

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.scatter(S_t_sne[original_instance_index, 0], S_t_sne[original_instance_index, 1], c='green', alpha=0.75, edgecolors='w', label='Original Instance')  # assuming the first point is the original instance
    
    # Highlight contrafactual instances if present
    if hasContrafactual:
        # Change color for the last k instances
        contrafactual_points = S_t_sne[-k:]
        counterfactual_classes = y[-k:]
        plt.scatter(contrafactual_points[:, 0], contrafactual_points[:, 1], c=counterfactual_classes, cmap='viridis', alpha=0.5, edgecolors='red', linewidths=1.5, label='Counterfactuals')

    plt.colorbar(scatter)
    plt.title('t-SNE visualization')
    if hasContrafactual:
        plt.legend('with contrafactuals')
    plot_path = os.path.join(test_folder, path)
    plt.savefig(plot_path)
    plt.close()

def plot_TSNE2(X, y, perpl, contrafactuals, original_instance_index, model, test_folder, path):
    X_scaled = normalize_if_needed(X)
    
    t_sne = manifold.TSNE(
        init = 'pca',
        n_components=2, #make it not hard coded
        perplexity=perpl,
        n_iter=2000,
        random_state=0
    )

    S_t_sne = t_sne.fit_transform(X_scaled)
    print(t_sne.kl_divergence_)

    if not pd.api.types.is_numeric_dtype(y):
        y = pd.Categorical(y).codes

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(S_t_sne[:, 0], S_t_sne[:, 1], c=y, cmap='viridis', alpha=0.5)
    plt.scatter(S_t_sne[original_instance_index, 0], S_t_sne[original_instance_index, 1], c='green', alpha=0.75, edgecolors='w', label='Original Instance')  # assuming the first point is the original instance
    
    # Get contrafactual classes
    counterfactual_classes = model.predict(contrafactuals)
    if not pd.api.types.is_numeric_dtype(counterfactual_classes):
        counterfactual_classes = pd.Categorical(counterfactual_classes).codes

    # Transform the counterfactuals using the fitted t-SNE model
    all_data = np.vstack([X_scaled, normalize_if_needed(contrafactuals)])
    new_instances_transformed = t_sne.fit_transform(all_data)[-len(contrafactuals):]

    # Plot the counterfactuals in "X" format
    plt.scatter(new_instances_transformed[:, 0], new_instances_transformed[:, 1], c=counterfactual_classes, cmap='viridis', alpha=0.5, edgecolors='red', linewidths=1.5, label='Counterfactuals')

    plt.colorbar(scatter)
    plt.title('t-SNE2 visualization')
    plot_path = os.path.join(test_folder, path)
    plt.savefig(plot_path)
    plt.close()


# for Graph
def get_edges(kernel):
    edges = []

    n_samples = kernel.shape[0]
    for i in range(n_samples):
        for j in range(n_samples):
            if kernel[i, j] != 0 :
                edges.append([i, j, kernel[i, j]])
    return edges


# for getting recommendations from path
def get_path_changes(X, path):
    points_in_path = deepcopy(path)
    features_over_time = []
    features_final_direction = get_features_direction(X[points_in_path[0]], X[points_in_path[-1]]) # 1 = increased ; -1 = decreased ; 0 = stay the same #here between original_ind and contrafactual
    features_range = get_features_range(X[points_in_path[0]], X[points_in_path[-1]])
    features_over_time.append(X[points_in_path[0]])
    points_in_path.pop(0)

    for point in points_in_path:
        features = X[point]
        features_over_time.append(analyze_features(features, features_final_direction, features_over_time, features_range))

    return features_over_time

def analyze_features(features, features_final_direction, features_over_time, features_range):
    result = []
    features_current_direction = get_features_direction(features_over_time[-1], features)
    for feature_idx in range(len(features)):
        if features_current_direction[feature_idx] == features_final_direction[feature_idx] and is_in_range(features_range, features, feature_idx):
            result.append(features[feature_idx])
        else:
            result.append(features_over_time[-1][feature_idx])

    return result

def get_features_direction(first, last):
    direction = []
    for feature in range(len(first)):
        if first[feature] > last[feature]:
            direction.append(-1)
        elif first[feature] < last[feature]:
            direction.append(1)
        else:
            direction.append(0)

    return direction

def get_features_range(first, last):
    ranges = []
    for feature in range(len(first)):
        start, end = sorted([first[feature], last[feature]])
        ranges.append([start, end])
    return ranges

def is_in_range(ranges, features, idx):
    return ranges[idx][0] <= features[idx] <= ranges[idx][-1]



# for normalizing data and datapoints before calculating its density
def is_normalized(data, feature_range=(0, 1)):
    min_val, max_val = feature_range
    return np.all(data >= min_val) and np.all(data <= max_val)

def normalize_if_needed(data):
    # Check if data is a single instance and reshape accordingly
    is_single_instance = False
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
        is_single_instance = True

    if not is_normalized(data):
        scaler = preprocessing.MinMaxScaler()
        data = scaler.fit_transform(data)
    
    # Reshape back to single instance if necessary
    if is_single_instance:
        data = data.flatten()
        
    return data



class density(object):
    def __init__(self, predictor, weight_function = None, 
        distance_threshold=None,
        density_threshold=None,
        # prediction_threshold= None,
        # howmanypaths = None,
        undirected = False):
        if not hasattr(predictor, 'predict_proba'):
            raise ValueError('Predictor needs to have attribute: \'predict proba\'')
        else:
            self.predictor = predictor

        if weight_function is None:
            self.weight_function = lambda x: -np.log(x)
        else:
            self.weight_function = weight_function

        self.undirected = undirected

        if density_threshold is None:
            self.density_threshold = 1e-5
        else:
            self.density_threshold = density_threshold

        if distance_threshold is None:
            self.distance_threshold = 12000 #change
        else:
            self.distance_threshold = distance_threshold

        # if prediction_threshold is None:
        #     self.prediction_threshold = 0.60    
        # else:
        #     self.prediction_threshold = prediction_threshold

        # if howmanypaths is None:
        #     self.howmanypaths = 5
        # else:
        #     self.howmanypaths = howmanypaths

    def generateSpaceDensity(self, X, y, f):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        if self.n_samples != self.y.shape[0]:
            raise ValueError('Inconsistent dimensions')
        self.predictions = self.predictor.predict_proba(X)
        self.get_kde(f)

    def get_kde(self, f):

        # FOR MULTIDIMENSIONAL DATASET
        np_first = -1
        np_second = 1
        qnt = 100
        scaler = preprocessing.MinMaxScaler()
        X_scaled = scaler.fit_transform(self.X)

        bandwidths = np.logspace(np_first, np_second, qnt)  # Adjusting bandwidth range
        f.write("\n\nnp.logspace( " + str(np_first) + ", " + str(np_second) + ", " + str(qnt) + " )")
        kernels = ['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine']

        
        grid = GridSearchCV(KernelDensity(), {'kernel': kernels, 'bandwidth': bandwidths} 
                                #cv=3
                                )
        grid.fit(X_scaled)

        # Visualization of density with the selected bandwidth
        self.density_estimator = grid.best_estimator_
        f.write("\n\nBest kernel:" + str(grid.best_params_['kernel']))
        f.write("\nBest bandwidth:" + str(grid.best_params_['bandwidth']) + "\n")
        log_densities = self.density_estimator.score_samples(X_scaled)
        densities = np.exp(log_densities)
        min_density_index = densities.argmin()
        min_density_value = densities[min_density_index]
        # min_density_instance = self.X.iloc[min_density_index]
        # print("here")

            # sns.kdeplot(data=self.X, bw_adjust=kde.bandwidth, kernel=kernel)
            # plt.title(f'Kernel: {kernel} with Bandwidth: {kde.bandwidth}')
            # plt.show()

        # # FOR OFICIAL FACE 2D DATASET
        # bandwidths = np.logspace(-2, 0, 20)  
        # #bandwidths = [0.65]
        # grid = GridSearchCV(
        #     KernelDensity(kernel='gaussian'),
        #     {'bandwidth': bandwidths},
        #     # cv=4,
        # )    
        # grid.fit(deepcopy(self.X))
        # self.density_estimator = grid.best_estimator_

        # 
        # f.write("\n\nBest kernel: gaussian")
        # f.write("\nBest bandwidth:" + str(grid.best_params_['bandwidth']) + "\n")

    def get_kde_density_2points(self, ind, original_ind, mode):
        dens=0
        density_scorer = self.density_estimator.score_samples
        
        #v0 = ind.reshape(-1, 1)
        #v1 = original_ind.reshape(-1, 1)
            #CHECK WHAT THIS CONDITIONS IS ABOUT
            #if not self.check_conditions(v0, v1):
                #continue
        dist = distance.euclidean(ind, original_ind)
        midpoint = normalize_if_needed((ind + original_ind)/2)
        density = density_scorer(midpoint.reshape(1, -1))
        # print("\nKDE Density between 2 points:\n")
        # if mode == 1:
            # dens = self.weight_function(np.exp(density)) * dist
        # else:
            # dens = self.weight_function(sigmoid(density)) * dist
        dens = (np.exp(density)) * dist

        # print("Density = " + str(density) + " np.exp(" + str(density) + ") = " + str(np.exp(density)) + " |  -log(" + str(np.exp(density)) + ") = " + str(dens[0]))

        # print("Density-Distance Tradeoff Value (dens):", dens)
        # print("Type of dens:", type(dens))
        # if isinstance(dens, np.ndarray):
        #     print("Shape of dens:", dens.shape)
        # else:
        #     print("dens is a scalar and does not have a shape.")
        
        return dens[0]

    def get_kde_density_population(self, population, mode, f):
        dens=[]
        density_scorer = self.density_estimator.score_samples
        population_normalized = normalize_if_needed(population)
        
        #v0 = ind.reshape(-1, 1)
        #v1 = original_ind.reshape(-1, 1)
            #CHECK WHAT THIS CONDITIONS IS ABOUT
            #if not self.check_conditions(v0, v1):
        #         #continue
        for ind in population_normalized:
            density = density_scorer(ind.reshape(1, -1))
            # if mode == 1:
            #     dens.append(self.weight_function(np.exp(density)))
            # else:
            #     dens.append(self.weight_function(sigmoid(density)))
            dens.append(np.exp(density))
            # f.write("Raw Density = " + str(density) + " np.exp(" + str(density) + ") = " + str(np.exp(density)) + " |  -log(" + str(np.exp(density)) + ") = " + str(dens[-1]) + "\n")
        
        return dens
    
    def get_kde_density_point(self, ind, mode):
        dens=0
        density_scorer = self.density_estimator.score_samples
        
        #v0 = ind.reshape(-1, 1)
        #v1 = original_ind.reshape(-1, 1)
            #CHECK WHAT THIS CONDITIONS IS ABOUT
            #if not self.check_conditions(v0, v1):
                #continue
        density = density_scorer(normalize_if_needed(ind).reshape(1, -1))
        # print("\nKDE Density on one point:\n")
        # if mode == 1:
        #     dens = self.weight_function(np.exp(density))
        # else:
        #     dens = self.weight_function(sigmoid(density))
        dens = (np.exp(density))

        # print("Density = " + str(density) + " np.exp(" + str(density) + ") = " + str(np.exp(density))+ " |  -log(np.exp(" + str(density) + ")) = " + str(dens))
        
        return dens[0]

    def getGraph(self):
        self.kernel = self.get_kernel()
        self.fit_graph()

    def get_kernel(self):
        density_scorer = self.density_estimator.score_samples
        kernel = self.get_weights_kde(density_scorer, 1)
        self.kernel = kernel
        return kernel

    def get_weights_kde(self, density_scorer,
                    mode):
        k = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            v0 = self.X[i, :].reshape(-1, 1)
            for j in range(self.n_samples):#range(i):
                v1 = self.X[j, :].reshape(-1, 1)
                # if not self.check_conditions(v0, v1):
                #     continue
                dist = np.linalg.norm(v0 - v1) #euclidian distance
                if dist <= self.distance_threshold:
                    midpoint_scaled = normalize_if_needed((v0 + v1)/2)
                    density = density_scorer(midpoint_scaled.reshape(1, -1))
                    if mode == 1:
                        k[i, j] = self.weight_function(np.exp(density)) * dist
                    else:
                        k[i, j] = self.weight_function(sigmoid(density)) * dist
                else:
                    k[i, j] = 0
        return k
    
    def fit_graph(self):

        def print_edges(edges):
            f = open("output.txt", "a")
            f.write("\n\n\n EDGES \n\n\n" + str(edges))
            for edge in edges:
                self.graph.add_edge(*edge)
            f.write("\n\n") 

        self.graph = Graph(undirected=self.undirected)
        edges = get_edges(self.kernel)
        for edge in edges:
            self.graph.add_edge(*edge)
        # print_edges(edges)
    
    def compute_path(
        self, 
        starting_point, #index (to use instance, change code to create starting_point_index)
        counterfactuals, #index (to use instance, change code to create counterfactuals_indexes)
        f
        ):
        dist, paths = dijsktra_tosome(deepcopy(self.graph), starting_point, counterfactuals)           
            
        recommendations = self.get_recommendations(paths)
            
        return paths, recommendations
    
    def get_recommendations(self, all_paths):
        counter = 0
        self.path_changes = []
        for path_id, path in all_paths.items():
            self.path_changes.append(get_path_changes(self.X, path))
            counter += 1

        return self.path_changes
    
    def print_recommendation(self, paths, recommendations, f):
        # f = open("path_recommendation.txt", "w")
        f.write("\n\n ############## FINAL RECOMMENDATIONS ############## \n\n")
        i=0
        for recom, (key, path) in zip(recommendations, paths.items()):
            f.write("------- PATH " + str(i) + " = " + str(path) + " -------- \n")
            for point in recom:
                f.write("POINT: ")
                for feature in point:
                    f.write("Feature: " + str(feature) + " | ")
                f.write("\n")
            f.write("\n\n")
            i = i + 1
